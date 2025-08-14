# Proyek Klasifikasi Harga Handphone dengan TFX Pipeline

- Nama: Moch Rifky Aulia Adikusumah
- Username Dicoding: rifkyadi

---

## Detail Proyek

| Deskripsi | Keterangan |
| :--- | :--- |
| Dataset | [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) - Dataset ini berisi data spesifikasi handphone dan targetnya adalah `price_range` (0=murah, 1=sedang, 2=mahal, 3=sangat mahal). |
| Masalah | Banyaknya model handphone baru dengan berbagai spesifikasi membuat konsumen dan produsen sulit untuk menentukan posisi harga yang tepat. Kesalahan dalam penentuan harga dapat menyebabkan produk tidak laku atau keuntungan yang tidak maksimal. |
| Solusi Machine Learning | Membangun sebuah model klasifikasi yang dapat memprediksi rentang harga (`price_range`) sebuah handphone berdasarkan fitur-fitur teknisnya (seperti RAM, kapasitas baterai, resolusi kamera, dll.). Solusi ini diimplementasikan dalam sebuah pipeline TFX yang otomatis, mulai dari data ingestion, validasi, preprocessing, training, evaluasi, hingga siap untuk deployment. Target utama adalah mencapai akurasi klasifikasi di atas 90% pada data validasi. |
| Metode Pengolahan Data | Menggunakan komponen `Transform` dari TFX. Fitur-fitur numerik (seperti `battery_power`, `ram`, `px_width`, dll.) akan dinormalisasi menggunakan standard scaling (Z-score) untuk menyamakan skala dan membantu model Konvergen lebih cepat. Fitur kategorikal (seperti `four_g`, `dual_sim`, dll.) yang sudah dalam format biner (0 atau 1) akan langsung digunakan. |
| Arsitektur Model | Menggunakan model Deep Neural Network (DNN) yang dibangun dengan TensorFlow/Keras. Arsitektur model terdiri dari: Beberapa layer `Dense` dengan fungsi aktivasi ReLU, sebuah layer `Dropout` untuk mencegah overfitting, layer output `Dense` dengan 4 unit (sesuai jumlah kelas `price_range`) dan fungsi aktivasi Softmax untuk klasifikasi multi-kelas. |
| Metrik Evaluasi | Metrik utama yang digunakan adalah Sparse Categorical Accuracy. Akurasi dipilih karena distribusi kelas pada dataset ini seimbang, sehingga akurasi dapat menjadi representasi performa model yang baik. |
| Performa Model | Model yang telah dilatih dan dievaluasi menggunakan TFX Evaluator berhasil mencapai akurasi sebesar 89% pada set data evaluasi. |
| Opsi deployment | Deksripsi tentang opsi deployment |
| Web app | Tautan web app yang digunakan untuk mengakses model serving. Contoh: [nama-model](https://model-resiko-kredit.herokuapp.com/v1/models/model-resiko-kredit/metadata)|
| Monitoring | Deksripsi terkait hasil monitoring dari model serving |