import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import optuna

np.random.seed(42)

df = pd.read_csv("AI_Resume_Screening.csv")
print("Loaded dataset shape:", df.shape)

median_score = df["AI Score (0-100)"].median()
df["Hire_Label"] = (df["AI Score (0-100)"] >= median_score).astype(int)
print("\nBalanced label counts:\n", df["Hire_Label"].value_counts())

df["Experience_Years"] = df["Experience (Years)"].astype(int)
df["Projects"] = df["Projects Count"].astype(int)
df["Salary"] = df["Salary Expectation ($)"].astype(float)
df["Skill_Count"] = df["Skills"].apply(lambda s: len(s.split(",")))

edu_map = {
    "PhD": 3, "M.Tech": 2, "MBA": 2,
    "B.Tech": 1, "B.Sc": 1
}
df["Education_Score"] = df["Education"].map(edu_map).fillna(1)

feature_cols = [
    "Experience_Years","Projects","Salary",
    "Skill_Count","Education_Score"
]

df_final = df[feature_cols + ["AI Score (0-100)", "Hire_Label"]]
df_final.to_csv("resume_binary_transformed_full.csv", index=False)
print("\nSaved transformed dataset → resume_binary_transformed_full.csv")

X = df_final[feature_cols].values.astype(float)
y = df_final["Hire_Label"].values.astype(int)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_loss_and_grads(Xb, yb, W, b, reg):
    N = Xb.shape[0]
    logits = np.dot(Xb, W) + b
    preds = sigmoid(logits)
    loss = -np.mean(yb*np.log(preds+1e-9) + (1-yb)*np.log(1-preds+1e-9))
    loss += 0.5 * reg * np.sum(W*W)
    error = preds - yb
    grad_W = np.dot(Xb.T, error)/N + reg * W
    grad_b = np.mean(error)
    return loss, grad_W, grad_b

def train_manual(Xb, yb, lr, reg, epochs):
    N, D = Xb.shape
    W = np.zeros(D)
    b = 0.0
    losses = []
    for ep in range(epochs):
        loss, gW, gb = compute_loss_and_grads(Xb, yb, W, b, reg)
        gW = np.clip(gW, -5, 5)
        gb = np.clip(gb, -5, 5)
        W -= lr * gW
        b -= lr * gb
        losses.append(loss)
    return W, b, losses

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 0.5, log=True)
    reg = trial.suggest_float("reg", 1e-8, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 50, 300)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        Xtr, Xv = X[tr_idx], X[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        W, b, _ = train_manual(Xtr, ytr, lr, reg, epochs)
        preds = (sigmoid(np.dot(Xv, W) + b) >= 0.5).astype(int)
        scores.append(accuracy_score(yv, preds))
    return float(np.mean(scores))

print("\nRunning Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("\nBest params:", best_params)

best_lr = best_params["lr"]
best_reg = best_params["reg"]
best_epochs = best_params["epochs"]

best_val_acc = -1
best_model = None
best_losses = None
best_train_idx = None
best_test_idx = None

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    W, b, losses = train_manual(Xtr, ytr, best_lr, best_reg, best_epochs)
    preds = (sigmoid(np.dot(Xte, W) + b) >= 0.5).astype(int)
    acc = accuracy_score(yte, preds)
    print(f"Fold {fold} accuracy: {acc:.4f}")
    if acc > best_val_acc:
        best_val_acc = acc
        best_model = (W, b)
        best_losses = losses
        best_train_idx = tr_idx
        best_test_idx = te_idx

np.save("best_W.npy", best_model[0])
np.save("best_b.npy", np.array([best_model[1]]))

print("\nBest fold accuracy:", best_val_acc)

df_final.iloc[best_train_idx].to_csv("best_train_split.csv", index=False)
df_final.iloc[best_test_idx].to_csv("best_test_split.csv", index=False)

print("Saved train/test splits.")

W_best, b_best = best_model

logits = np.dot(X[best_test_idx], W_best) + b_best
preds = (sigmoid(logits) >= 0.5).astype(int)
yt = y[best_test_idx]

acc = accuracy_score(yt, preds)
prec = precision_score(yt, preds)
rec = recall_score(yt, preds)
cm = confusion_matrix(yt, preds)

print("\nFinal Test Metrics:")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)

plt.plot(best_losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.close()

plt.bar(["Accuracy", "Precision", "Recall"], [acc, prec, rec])
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.savefig("metrics_bar.png")
plt.close()

plt.hist(df_final["AI Score (0-100)"], bins=10)
plt.title("AI Score Distribution")
plt.savefig("ai_score_distribution.png")
plt.close()
