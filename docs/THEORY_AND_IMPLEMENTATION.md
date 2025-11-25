````markdown
# 🏆 CS 4774 Machine Learning 🏆
University of Virginia - Computer Science  
Homework 3 — **Knowledge Distillation for AI Dermatologist**  
Instructor: Hadi Daneshmand | TA: Wenqian Ye (wenqian@virginia.edu)

---

# 📚 Knowledge Distillation: Theory and Implementation

## Introduction

Knowledge distillation is a model compression technique proposed by Hinton et al. (2015) that enables a compact "student" model to achieve performance comparable to a large, complex "teacher" model. The key insight is that the soft probability distributions produced by the teacher contain rich information about the relationships between classes, which is lost when using only hard labels.

In this homework, you will implement knowledge distillation to improve your skin disease classifier from Homework 1 by learning from **MedSigLIP**, a powerful medical imaging model from Google.

> 📖 **Required Reading:**
>
> **Hinton, G., Vinyals, O., & Dean, J. (2015).**  
> *Distilling the Knowledge in a Neural Network.*  
> *NIPS Deep Learning Workshop.*  
> <https://arxiv.org/abs/1503.02531>  
>
> This paper introduces the concept of knowledge distillation and the formulation we will use in this homework. Please read this paper to understand the theoretical foundation.

---

## Mathematical Formulation

> **Note:** The formulation below follows Hinton et al. (2015). Temperature scaling and softened targets are central to their method.

### 1. Temperature-Scaled Softmax

Given a model's logits  
\[
\mathbf{z} = [z_1, z_2, \dots, z_C]
\]  
for \(C\) classes, the standard softmax function is:
\[
p_i = \frac{\exp(z_i)}{\sum_{j=1}^C \exp(z_j)}.
\]

Hinton et al. introduce a **temperature parameter** \(T\) to soften the probability distribution:
\[
p_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^C \exp(z_j / T)}.
\]

When \(T = 1\), this reduces to the standard softmax. As \(T\) increases, the distribution becomes softer (less peaked), which reveals more information about similarities between classes.

### 2. Distillation Loss

Let \(\mathbf{z}^T\) and \(\mathbf{z}^S\) denote the logits of the **teacher** and **student** models, respectively. Let \(y\) denote the ground-truth label, and \(T\) denote the temperature parameter. The distillation loss combines two components:

#### Component 1: Hard Label Loss

Standard cross-entropy between student predictions and true labels:
\[
\mathcal{L}_{\text{hard}} = -\sum_{i=1}^C \mathbf{1}[y = i] \log\left(\frac{\exp(z_i^S)}{\sum_{j=1}^C \exp(z_j^S)}\right)
= -\log \left(\frac{\exp(z_y^S)}{\sum_{j=1}^C \exp(z_j^S)}\right).
\]

Here \(\mathbf{1}[y = i]\) is the indicator function (1 if \(y = i\), 0 otherwise), and \(z_i^S\) is the \(i\)-th element of the student logits \(\mathbf{z}^S\). This is the usual cross-entropy loss between the student softmax and the one-hot encoded true label.

#### Component 2: Soft Label Loss (Distillation)

First define the temperature-scaled probabilities for teacher and student:
\[
p_i^T(T) = \frac{\exp(z_i^T / T)}{\sum_{j=1}^C \exp(z_j^T / T)}, \quad
p_i^S(T) = \frac{\exp(z_i^S / T)}{\sum_{j=1}^C \exp(z_j^S / T)}.
\]

The soft loss is the cross-entropy between the teacher and student soft targets, scaled by \(T^2\):
\[
\mathcal{L}_{\text{soft}} = -T^2 \sum_{i=1}^C p_i^T(T)\, \log p_i^S(T).
\]

Equivalently, this is proportional to the Kullback–Leibler divergence
\[
D_{\text{KL}}\big(p^T(T) \,\|\, p^S(T)\big)
\]
between the teacher and student soft targets (with the \(T^2\) factor compensating for the scaling effect of \(T\) on gradients).

#### Total Loss

The final loss is a weighted combination:
\[
\mathcal{L} = \alpha\, \mathcal{L}_{\text{hard}} + (1 - \alpha)\, \mathcal{L}_{\text{soft}},
\]
where \(\alpha \in [0,1]\) is a hyperparameter controlling the balance between hard labels and soft targets. Typical values: \(\alpha \in [0.1, 0.5]\).

### 3. Training Procedure

1. **Load teacher model**  
   Load pre-trained MedSigLIP, set to evaluation mode, and freeze all parameters (no training needed).

2. **Forward pass**  
   Compute teacher logits \(\mathbf{z}^T\) (with no gradients) and student logits \(\mathbf{z}^S\).

3. **Compute soft probabilities**  
   Apply temperature-scaled softmax to both teacher and student logits with temperature \(T\).

4. **Compute losses**  
   Calculate \(\mathcal{L}_{\text{hard}}\) and \(\mathcal{L}_{\text{soft}}\).

5. **Backpropagate**  
   Update **only student model parameters** using total loss \(\mathcal{L}\) (teacher remains frozen).

---

## The Teacher Model: MedSigLIP from Google

For this homework, we use **MedSigLIP** (Medical SigLIP) from Google as the teacher model. SigLIP (Sigmoid Loss for Language-Image Pre-training) is a variant of CLIP that uses sigmoid loss instead of softmax, making it more efficient for large-scale vision–language pretraining.

MedSigLIP is specifically trained on medical imaging datasets, including **chest X-rays, dermatology images, ophthalmology images, and histopathology slides**, making it particularly well-suited for dermatology tasks. It has learned rich visual representations from diverse medical images.

> 📚 **Resources:**
>
> - **Model:**  
>   <https://huggingface.co/google/medsiglip-448>  
> - **Documentation:**  
>   <https://developers.google.com/health-ai-developer-foundations/medsiglip>  
> - **Paper:**  
>   Sellergren, A., et al. (2025). *MedGemma Technical Report.* arXiv:2507.05201

> ⚠️ **Important Note:**  
> You will use the pre-trained MedSigLIP model **as-is for inference only** (no fine-tuning needed). Load the model, set it to evaluation mode, and use it to generate soft targets for your student model. The model accepts 448×448 images and contains a 400M parameter vision encoder. Your focus should be on designing and training a compact student model (<25 MB) that learns from MedSigLIP's predictions.

---

## Key Hyperparameters

| Parameter        | Symbol | Typical Range | Description                                     |
|-----------------|--------|---------------|-------------------------------------------------|
| Temperature     | \(T\)  | 2 – 10        | Controls softness of probability distributions  |
| Balance factor  | \(\alpha\) | 0.1 – 0.5  | Weight for hard label loss                      |
| Learning rate   | \(\eta\)   | 1e-4 – 1e-3 | Student model learning rate                     |

*Note: These are starting points. Experimentation is key to finding optimal values for your specific architecture.*

---

## Implementation Tips

### 💡 Tip 1: Use pre-trained teacher as-is

Load the pre-trained MedSigLIP model and use it directly for inference (no training needed). Set it to evaluation mode and use `torch.no_grad()` when computing teacher predictions to save memory and computation.

### 💡 Tip 2: Experiment with temperature

Higher temperatures (5–10) work well when teacher and student architectures are very different. Lower temperatures (2–4) work when they're similar.

### 💡 Tip 3: Balance is crucial

Start with \(\alpha = 0.3\). If student accuracy is poor, increase \(\alpha\) to rely more on hard labels. If the student is close to the teacher but not improving, decrease \(\alpha\).

### 💡 Tip 4: Monitor both losses

Track \(\mathcal{L}_{\text{hard}}\) and \(\mathcal{L}_{\text{soft}}\) separately during training to understand whether your student is learning more from labels or from the teacher.

### 💡 Tip 5: Focus on student architecture

Your main task is designing a compact student model (<25 MB) that learns effectively from the teacher. We recommend starting with **ShuffleNetV2** (~5 MB), which leaves plenty of room for experimentation. Explore different architectures, depth, width, and regularization techniques.

---

## Model Submission Instructions

The submission process is identical to Homework 1. Follow these steps:

### Step 1: Save Your Student Model

Only submit your student model (not the teacher):

```python
student_model.eval()
scripted_model = torch.jit.script(student_model)
scripted_model.save("student_model.pt")
print("Model saved")
```

### Step 2: Submit to Server

```python
import requests

def submit_model(token: str, model_path: str, 
                 server_url="http://hadi.cs.virginia.edu:8000"):
    with open(model_path, "rb") as f:
        files = {"file": f}
        data = {"token": token}
        response = requests.post(f"{server_url}/submit", 
                                 data=data, files=files)
        resp_json = response.json()
        if "message" in resp_json:
            print(f"✅ {resp_json['message']}")
        else:
            print(f"❌ Submission failed: {resp_json.get('error')}")

# Use your token from Homework 1
my_token = "your_token_here"
submit_model(my_token, "student_model.pt")
```

### Step 3: Check Status

```python
import requests

def check_submission_status(my_token):
    url = f"http://hadi.cs.virginia.edu:8000/submission-status/{my_token}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return
    
    attempts = response.json()
    for a in attempts:
        score = f"{a['score']:.4f}" if isinstance(a['score'], (float, int)) else "None"
        model_size = f"{a['model_size']:.4f}" if isinstance(a['model_size'], (float, int)) else "None"
        
        print(
            f"Attempt {a['attempt']}: "
            f"Score={score}, "
            f"Model size={model_size}, "
            f"Status={a['status']}"
        )

check_submission_status(my_token)
```

---

## Rules & Constraints

- Use the same token from Homework 1  
- ⏱ Submit once per 15 minutes  
- 📊 Model must be < 25 MB  
- ✅ Only the student model should be submitted  
- 📚 You may use any architecture for the student model (ShuffleNetV2 recommended as starting point, ~5 MB)  
- 🎓 The teacher model (MedSigLIP from Google) should only be used during training  

---

## Leaderboard

🥇 Check your ranking on the **Homework 3 Leaderboard**:  
<http://hadi.cs.virginia.edu:8000/leaderboard3>

(This is a separate leaderboard for HW3. Same evaluation metrics as HW1, but different submissions.)

---

## Starter Code

Download the starter notebook with implementation template:  
**Download Starter Notebook:** <http://hadi.cs.virginia.edu:8000/download/starter-hw3>

> **CHALLENGE:**  
> Can you surpass your Homework 1 score? How much improvement can distillation provide?

---

## 📚 References and Further Reading

- **Knowledge Distillation (Required):**  
  Hinton, G., Vinyals, O., & Dean, J. (2015).  
  *Distilling the Knowledge in a Neural Network.*  
  *NIPS Deep Learning Workshop.*  
  <https://arxiv.org/abs/1503.02531>  
  *This is the foundational paper that introduces knowledge distillation. All mathematical formulations in this homework follow this paper.*

- **MedSigLIP Model:**  
  Sellergren, A., et al. (2025).  
  *MedGemma Technical Report.*  
  *arXiv preprint arXiv:2507.05201.*  
  - Model: <https://huggingface.co/google/medsiglip-448>  
  - Documentation: <https://developers.google.com/health-ai-developer-foundations/medsiglip>

- **SigLIP:**  
  Zhai, X., et al. (2023).  
  *Sigmoid Loss for Language Image Pre-Training.*  
  *ICCV 2023.*  
  *Background on the SigLIP architecture that MedSigLIP is based on.*

- **Feature distillation (Advanced):**  
  Romero, A., et al. (2015).  
  *FitNets: Hints for Thin Deep Nets.*  
  *ICLR 2015.*  
  *For students interested in exploring feature-based distillation beyond the standard approach.*
````
