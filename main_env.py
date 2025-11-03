import os
import time
import optuna
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from RubikCube_env import RubikCube


# ==========================================
# ENVIRONMENT CREATION
# ==========================================
def make_env():
    return RubikCube(size=3, difficulty_level=3, render_mode=None)


# ==========================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ==========================================
def objective(trial):
    # --- Choose algorithm ---
    algo_name = trial.suggest_categorical("algo", ["PPO", "A2C"])

    # --- Common hyperparameters ---
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.8)

    # --- PPO-specific params ---
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10])

    # --- Print trial start info ---
    print(f"\n=== Starting trial {trial.number} ===")
    print(f"Algorithm: {algo_name}")
    print(f"learning_rate={learning_rate:.6f}, gamma={gamma:.4f}, gae_lambda={gae_lambda:.4f}")
    print(f"ent_coef={ent_coef:.6f}, vf_coef={vf_coef:.4f}")
    if algo_name == "PPO":
        print(f"n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}")
    else:
        print("Using A2C (no batch_size/n_epochs)")

    # --- Create environment ---
    env = VecMonitor(DummyVecEnv([make_env]))

    # --- Create per-trial log directory ---
    log_dir = f"tensorboard_logs/rubik_tune/trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)

    # --- Instantiate model ---
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            tensorboard_log=log_dir,
            device="cpu",
        )
    else:  # A2C
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            tensorboard_log=log_dir,
            device="cpu",
        )

    # --- Custom logger for each trial ---
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # --- Train briefly ---
    model.learn(total_timesteps=50_000, tb_log_name=f"trial_{trial.number}")

    # --- Evaluate ---
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)

    # --- Print trial result ---
    print(f"Trial {trial.number} finished → mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward


# ==========================================
# OPTUNA STUDY SETUP
# ==========================================
if __name__ == "__main__":
    study_name = "rubik_tune"
    storage_name = f"sqlite:///{study_name}.db"

    # Load or create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )

    # Optimize
    study.optimize(objective, n_trials=8, n_jobs=1)  # reduce n_trials for testing

    print("\n✅ Optimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Reward: {trial.value}")
    print(f"  Params: {trial.params}")

    # ==========================================
    # TRAIN BEST MODEL
    # ==========================================
    best_algo = trial.params["algo"]
    env = VecMonitor(DummyVecEnv([make_env]))

    # Filter only valid params for each algorithm
    params = {k: v for k, v in trial.params.items() if k != "algo"}
    if best_algo == "PPO":
        best_model = PPO("MlpPolicy", env, **params, verbose=1, device="cpu")
    else:
        # Remove PPO-only params for A2C
        a2c_params = {k: v for k, v in params.items() if k not in ["batch_size", "n_epochs"]}
        best_model = A2C("MlpPolicy", env, **a2c_params, verbose=1, device="cpu")

    print(f"\n🚀 Training best model ({best_algo}) for 1k timesteps...")
    best_model.learn(total_timesteps=150_000)

    os.makedirs("best_models", exist_ok=True)
    best_model.save(f"best_models/{best_algo}_best_model.zip")
    print(f"✅ Best model saved: best_models/{best_algo}_best_model.zip")

    # ==========================================
    # PRINT ALL TRIALS IN DATABASE
    # ==========================================
    print("\n=== All trials in Optuna database ===")
    loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
    for t in loaded_study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")