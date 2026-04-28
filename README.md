# 🚀 Clifford-Dirac 3D PartNeXt Segmentation (Pure 3D)

Bu depo, [PartNeXt](https://huggingface.co/datasets/AuWang/PartNeXt) veri seti üzerinde **Class-Agnostic (Kategori Bağımsız)** 3D parça segmentasyonu yapmak için geliştirilmiş, **Clifford Cebiri (Clifford Algebra) ve Dirac Operatörlerini** temel alan bir derin öğrenme modelini içermektedir.

Geleneksel 2D destekli (Knowledge Distillation) veya kaba hiyerarşili modellerin aksine, bu model tamamen **"Safkan 3D" (Pure 3D)** geometrisine dayanır. Model, nesnenin ne olduğunu bilmeden sadece nokta bulutlarının (x, y, z) geometrik özelliklerinden yola çıkarak **315 farklı ince detayı (fine-grained parts)** ayırt edebilir.

## ✨ Öne Çıkan Özellikler

* 🏆 **Pure 3D SOTA:** 2D Foundation (SAM vb.) modellerinden destek (kopya) almadan, sadece saf 3D geometri ile **%32.04 mIoU** elde ederek P3-SAM ve SAMPart3D gibi devleri geride bırakır.
* 🧠 **Ultra-Hafif ve Verimli:** Devasa hibrit modellerin 600M+ parametresine kıyasla sadece **~1.25 Milyon parametre** ile SOTA performansına ulaşır (`base_channels=32`).
* 🧮 **Triton Destekli Clifford Çarpımı:** GPU üzerinde bivektör ve pseudo-skaler işlemlerinin ultra hızlı hesaplanması için özel yazılmış Triton kernel'ları içerir.
* 🛡️ **Sayısal Kararlılık:** Clifford uzayındaki gradyan patlamalarına (NaN problemleri) karşı optimize edilmiş korumalı (`eps=1e-4`, `clamp`, `tanh`) mimari.
* 💻 **CPU / HuggingFace Uyumlu Çıkarım:** Sadece tahmin (inference) yapmak isteyenler için Triton gerektirmeyen, saf PyTorch ile yazılmış `model.py` yedeği ile Gradio UI sunar.

## 📊 Performans Karşılaştırması (PartNeXt Sınıf-Bağımsız)

| Model | Yöntem Tipi | Parametre (Yaklaşık) | mIoU (Inst.) |
| :--- | :---: | :---: | :---: |
| PartField | Hybrid (2D Distillation) | > 500M | 41.50 |
| **Clifford-Dirac (Ours)** | **Pure 3D (From Scratch)** | **~ 1.25M** | **32.04** 👑 |
| P3-SAM | Hybrid (2D Distillation) | > 600M | 31.94 |
| SAMPart3D | Pure 3D | > 20M | 29.62 |

> **Not:** Hibrit modeller (PartField, P3-SAM), Meta'nın 2D Segment Anything (SAM) görsel hafızasını kullanarak 3D veriye etiket giydirir. Modelimiz ise sınıflandırıcı tüyo (category embed) veya 2D ön eğitim (pre-train) kullanmadan, bu başarıyı sadece *Clifford Bivektörleri* üzerinden geometrik öğrenmeyle elde etmiştir.

---

## 🛠️ Kurulum

Python 3.10+ ve CUDA destekli bir sistem önerilmektedir. 

## 💻 HF Space

https://huggingface.co/spaces/yusuf-tiryaki/PartNeXt
