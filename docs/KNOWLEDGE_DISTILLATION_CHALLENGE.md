# 🏆 CS 4774 Machine Learning 🏆
University of Virginia - Computer Science  
Homework 3 — **Knowledge Distillation for AI Dermatologist**  
Instructor: Hadi Daneshmand | TA: Wenqian Ye (wenqian@virginia.edu)  
Deadline: December 5, 2025, 8:00 PM

## Welcome to your third ML adventure!
## Knowledge Distillation Challenge
In Homework 1, you built a compact image classifier for 10 skin conditions. Now, in Homework 3, you'll take your model to the next level using **knowledge distillation** — a powerful technique that allows a small "student" model to learn from a large, pre-trained "teacher" model.

**The Challenge:**  
Can you improve your Homework 1 accuracy by leveraging the knowledge of a powerful medical imaging model?

**The Teacher:**  
You'll use **MedSigLIP** from Google, a state-of-the-art vision model pre-trained on medical images. This powerful teacher will guide your compact student model to achieve better performance than training from scratch.

**The Goal:**  
Build a student model that remains compact (<25 MB) but achieves significantly higher accuracy than your Homework 1 submission by learning from the teacher's soft predictions and rich feature representations.

---

## What is Knowledge Distillation?
Knowledge distillation is a model compression technique where a smaller "student" model learns to mimic the behavior of a larger "teacher" model. Rather than just learning from hard labels (e.g., class 0, 1, 2), the student learns from the teacher's soft probability distributions, which contain richer information about relationships between classes.

This technique was introduced by Hinton et al. (2015) in  
["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531).  
You'll implement this method using **MedSigLIP** from Google ([HuggingFace model](https://huggingface.co/google/medsiglip-448)) as your teacher model.

For detailed mathematical formulation and implementation guidance, see the **[Instructions](/instructions-hw3)** page.

---

## The Dataset: Same as Homework 1
You'll use the same 10,000 training images from Homework 1, and your model will be evaluated on the same test set with 10 skin disease classes:

- 1. Eczema — 751 images  
- 2. Melanoma — 938 images  
- 3. Atopic Dermatitis — 563 images  
- 4. Basal Cell Carcinoma — 993 images  
- 5. Melanocytic Nevi — 2381 images  
- 6. Benign Keratosis-like Lesions — 931 images  
- 7. Psoriasis / Lichen Planus — 921 images  
- 8. Seborrheic Keratoses / Benign Tumors — 828 images  
- 9. Tinea / Fungal infections — 763 images  
- 10. Warts / Viral Infections — 931 images  

---

## Your Mission
- Implement knowledge distillation using MedSigLIP (from Google) as the teacher model  
- Train a compact student model that learns from both ground-truth labels and teacher predictions  
- Achieve higher accuracy than your Homework 1 baseline  
- Student model must be **< 25 MB** on disk  

> 💡 **Recommended Starting Point:**  
> Consider using **ShuffleNetV2** as your student model architecture. It typically produces models **< 5 MB**, leaving plenty of room under the 25 MB limit.

---

## How Will You Be Judged?
- **Primary metric:** Weighted score = (10 × Macro F1-score) − (Model size factor).  
  F1 is weighted 10× more than model size.  
- **Tie-breaker:** If weighted scores are equal, the smaller model wins.  
- Submission limit: **1 submission per 15 minutes per team**  
- Model file size limit: **< 25 MB**  
- Only latest submission is saved  
- Test your model locally before submitting  

---

## Leaderboard & Evaluation
Homework 3 has a leaderboard for evaluation. Your submissions are evaluated on a test dataset stored securely on the server and **not directly accessible**.

- **[HW3 Leaderboard](http://hadi.cs.virginia.edu:8000/leaderboard3)** – View your evaluated model results.

**Key points:**
- Same training data as Homework 1  
- Distilled model must generalize to unseen test data  
- You receive leaderboard feedback after each submission  

---

## Instructions & Starter Code
We provide detailed math formulation, implementation guidance, and starter code:

**http://hadi.cs.virginia.edu:8000/instructions-hw3**

---

# Knowledge Distillation Concept
**Teacher Model (Large)**  
🧠  
Rich Knowledge  
⬇️  
**Student Model (Compact)**  
📱  
Efficient Performance  

Transfer knowledge from a powerful teacher model to create a compact student model that maintains high accuracy while being deployment-ready.

---

# Quick Actions
- [HW3 Leaderboard](/leaderboard3)  
- [Register (get token)](/register-page)  
- [Download Training dataset](/download/train-dataset)  
- [Instructions](/instructions-hw3)  
- [Download Starter Code](/download/starter-hw3)  

Questions? Email TA: [wenqian@virginia.edu](mailto:wenqian@virginia.edu)

---

# Good luck, future ML pioneers!
