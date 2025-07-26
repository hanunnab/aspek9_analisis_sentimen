import streamlit as st
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import torch
import gdown
import os

# === Variabel Global ===
# Mendefinisikan nama folder dan file untuk model
MODEL_ASPEK = 'aspek9'
URL_MODEL_ASPEK = '1KAsPdYAB-CvmJLK8OILmTGTVOSdikczt' #https://drive.google.com/file/d/1KAsPdYAB-CvmJLK8OILmTGTVOSdikczt/view?usp=drive_link
NAMA_MODEL_ASPEK = 'model_aspek9.pt'

MODEL_SENTIMEN = 'sentimen9'
URL_MODEL_SENTIMEN = '1696QFaz-TpienIW0aqRdv0uQTjK5fE9M' #https://drive.google.com/file/d/1696QFaz-TpienIW0aqRdv0uQTjK5fE9M/view?usp=drive_link
NAMA_MODEL_SENTIMEN = 'model_sentimen9.pt'

# === Fungsi-fungsi Aplikasi ===

def download_model(url, folder_path, filename):
    """
    Mengunduh file model dari Google Drive jika belum ada secara lokal.
    """
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    
    if os.path.exists(save_path):
        print(f"Model '{filename}' sudah ada.")
        return save_path

    print(f"Mengunduh model '{filename}'...")
    gdown.download(f"https://drive.google.com/uc?id={url}", save_path, quiet=False)
    print("Unduhan selesai.")
    return save_path

@st.cache_resource
def get_model_aspek():
    """
    Memuat model klasifikasi aspek.
    Menggunakan cache agar model hanya dimuat sekali.
    """
    path_aspek = os.path.join(MODEL_ASPEK, NAMA_MODEL_ASPEK)
    checkpoint = "indobenchmark/indobert-base-p1"
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.load_state_dict(torch.load(path_aspek, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def get_model_sentimen():
    """
    Memuat model klasifikasi sentimen.
    Menggunakan cache agar model hanya dimuat sekali.
    """
    path_sentimen = os.path.join(MODEL_SENTIMEN, NAMA_MODEL_SENTIMEN)
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = 3
    model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)
    model.load_state_dict(torch.load(path_sentimen, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_model_aspek(model, text):
    """
    Membuat prediksi aspek dari teks input.
    """
    tokenizer_aspek = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    inputs = tokenizer_aspek(
        text,
        return_tensors="pt",      
        truncation=True,           
        padding=True               
    )
    with torch.no_grad():
        outputs = model(**inputs)       
        logits = outputs.logits 
        hasil = torch.argmax(logits, dim=1).item()
    return hasil

def predict_model_sentimen(model, text):
    """
    Membuat prediksi sentimen dari teks input.
    """
    tokenizer_sentimen = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    inputs = tokenizer_sentimen(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        hasil = torch.argmax(outputs.logits, dim=1).item()
    return hasil
            
def main():
    """
    Fungsi utama untuk menjalankan aplikasi Streamlit.
    """
    # Mengatur konfigurasi halaman dan judul
    st.set_page_config(page_title="Analisis Sentimen", page_icon="🤖", layout="centered")

    # KODE CSS UNTUK TAMPILAN UI
    # Semua kode CSS didefinisikan di dalam string multi-baris ini.
    CUSTOM_THEME = """
    <style>
        
        /* Mengubah tampilan tombol utama */
        .stButton>button {
            border-radius: 10px;        /* Sudut yang membulat */
            color: #E5E1DA;             /* Warna teks tombol biru */
            background-color: #070F2B;  /* Latar belakang tombol putih */
            font-weight: bold;
            transition: all 0.2s;       /* Efek transisi halus saat hover */
            width: 100%;                /* Tombol memenuhi lebar kontainer */
        }

        /* Mengubah tampilan tombol saat kursor diarahkan ke atasnya */
        .stButton>button:hover {
            border-color: #070F2B;      /* Warna border lebih gelap */
            color: #FFFFFF;             /* Teks menjadi putih */
            background-color: #1976D2;  /* Latar belakang menjadi biru */
            transform: scale(1.02);     /* Sedikit membesar saat di-hover */
        }
        
        /* Mengubah tampilan area input teks */
        .stTextArea textarea {
            border: 1px solid #535C91;
            border-radius: 10px;
            background-color: #1B1A55;
            color: #FFFFFF;
        }

        /* Efek saat kursor mouse di atas area input teks */
        .stTextArea textarea:hover {
            border-color: #9DB2BF;      /* Warna border menjadi lebih terang */
            box-shadow: 0 0 5px #9DB2BF;  /* Memberi efek cahaya/glow di sekitar kotak */
}

        /* Mengubah tampilan kotak pesan (info, warning, success, error) */
        .stAlert {
            border-radius: 10px;
        }

        /* === PENGATURAN TEKS UI ANDA === */

    /* Mengatur st.title. Menggunakan selector lebih spesifik */
    [data-testid="stAppViewContainer"] h1 {
        font-family: 'Verdana', sans-serif;
        color: #D9EAFD; 
        text-align: center;
        padding-bottom: 35px; 
    }

    /* Mengatur label untuk st.text_area (tulisan "masukan teks...") */
    .stTextArea label {
        font-weight: bold;
        font-size: 1.2rem; /* Ukuran font 1.2 kali dari normal */
        color: #E5E1DA; /* Warna krem */
    }

    /* Mengatur semua kotak pesan (warning, success, error, info) */
    [data-testid="stAlert"] {
        border-radius: 10px;
        font-size: 1.05rem;
    }

    /* Penampung untuk tautan agar bisa diatur posisinya */
    .back-link-container {
        text-align: center;      /* Membuat tautan berada di tengah */
        margin-top: 100px;        /* Memberi jarak 50px dari elemen di atasnya */
        padding-bottom: 30px;    /* Memberi jarak di bagian bawah halaman */
    }

    /* Mengatur tampilan tautan (link) itu sendiri */
    .back-link-container a {
        color: #FBFBFB;           /* Warna teks abu-abu kebiruan yang lembut */
        text-decoration: none;     /* Menghilangkan garis bawah default */
        font-size: 0.8rem;         /* Sedikit memperbesar ukuran font */
        transition: all 0.2s;      /* Efek transisi yang mulus untuk hover */
    }

    /* Mengatur tampilan tautan saat kursor diarahkan ke atasnya */
    .back-link-container a:hover {
        text-decoration: underline; /* Menampilkan garis bawah saat di-hover */
        color: #FFFFFF;            /* Mengubah warna teks menjadi putih terang */
    }
    </style>
    """
    
    # Menerapkan CSS ke aplikasi
    st.markdown(CUSTOM_THEME, unsafe_allow_html=True)

    # Mengunduh model saat aplikasi pertama kali dijalankan
    download_model(URL_MODEL_ASPEK, MODEL_ASPEK, NAMA_MODEL_ASPEK)
    download_model(URL_MODEL_SENTIMEN, MODEL_SENTIMEN, NAMA_MODEL_SENTIMEN)

    # Memuat model ke dalam memori
    model_aspek = get_model_aspek()
    model_sentimen = get_model_sentimen()
    
    # === Tampilan Antarmuka (UI) ===
    st.title("Aspek Kebermanfaatan")
    user_input = st.text_area("Masukkan teks ulasan untuk dianalisis:", height=150)
    
    if st.button("Prediksi"):
        # Memastikan pengguna memasukkan teks sebelum prediksi
        if not user_input.strip():
            st.warning("Harap masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Sedang memproses..."):
                aspek = predict_model_aspek(model_aspek, user_input)
                if aspek == 0:
                    st.warning("Teks tidak terdeteksi sebagai ulasan aspek kebermanfaatan.")
                else:
                    sentimen = predict_model_sentimen(model_sentimen, user_input)
                    if sentimen == 0:
                        st.success("✅ Prediksi Sentimen: Positif")
                    elif sentimen == 2:
                        st.error("❌ Prediksi Sentimen: Negatif")
                    else: # Untuk sentimen netral atau lainnya
                        st.info("ℹ️ Prediksi Sentimen: Netral")

    # ✅ TAUTAN KEMBALI DITAMBAHKAN DI SINI
    st.markdown(
        """
        <div class="back-link-container">
            <a href="#" target="_self">⬅️ Kembali ke Landing Page</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
