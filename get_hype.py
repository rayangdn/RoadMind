import optuna

from train import train

def objective(trial):
    
    model, best_model_val_ade = train(
        num_epochs=100,
        lr=trial.suggest_loguniform("lr", 1e-4, 1e-3),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-5, 1e-4),
        scheduler_factor=trial.suggest_float("scheduler_factor", 0.6, 0.8),
        scheduler_patience=5,
        precision='high',
        hidden_dim=128,
        image_embed_dim=256,
        num_layers_gru=2,
        dropout_rate=trial.suggest_float("dropout_rate", 0.3, 0.5),
        weight_depth=trial.suggest_float("weight_depth", 7, 12),
        weight_semantic=trial.suggest_float("weight_semantic", 0.1, 0.3),
        use_depth_aux=True,
        use_semantic_aux=True,
        include_heading=False,
        include_dynamics=True,
        batch_size=64,
        logger_name='roadmind'
    )
    
    return best_model_val_ade

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
    
    import json
    with open('./logs/roadmind/best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save hyperparameters in a readable text format
    with open('./logs/roadmind/best_hyperparameters.txt', 'w') as f:
        f.write(f"Best ADE value: {study.best_value}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
if __name__ == "__main__":
    main()
    
    
    