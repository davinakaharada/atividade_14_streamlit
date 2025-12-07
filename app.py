import streamlit as st
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import numpy as np
# Importações necessárias para o reconhecimento facial e comparação
import face_recognition

# --- Configurações do MongoDB ---
# A URI com as credenciais
uri = 'mongodb+srv://davinakaharada_db_user:7l8hyTa7SQM9cvBG@cluster0.rno9hnk.mongodb.net/?appName=Cluster0'
client = MongoClient(uri)
db = client['midias']
fs = gridfs.GridFS(db)
# Coleção que armazena os embeddings faciais (Altere 'rostos' se o nome for diferente!)
rostos_collection = db['rostos']

# --- Funções Auxiliares de Reconhecimento Facial ---

@st.cache_resource
def detectar_e_codificar_face(imagem_bytes):
    """
    Detecta a face na imagem e retorna o embedding (codificação facial).
    """
    try:
        # 1. Converte bytes para array NumPy
        imagem = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
        img_array = np.array(imagem)
        
        # 2. Localiza as faces e calcula o embedding da primeira face
        face_locations = face_recognition.face_locations(img_array)
        
        if not face_locations:
            return None, "Nenhuma face detectada na imagem."

        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        
        if not face_encodings:
             return None, " Face detectada, mas falha ao gerar o embedding."

        return face_encodings[0], None
    
    except Exception as e:
        return None, f"Erro ao processar imagem: {e}"

def encontrar_face_mais_parecida(novo_embedding):
    """
    Compara o novo embedding com todos os armazenados no MongoDB
    e retorna a referência da face mais parecida.
    """
    # 1. Recupera todos os embeddings do MongoDB
    # Buscamos o embedding e o filename (que é a chave para o GridFS)
    dados_rostos = list(rostos_collection.find({}, {'embedding': 1, 'filename': 1}))
    
    if not dados_rostos:
        return None, 0.0, "Nenhum embedding armazenado para comparação."
        
    embeddings_conhecidos = []
    ids_rostos = [] # Usado para armazenar o 'filename' de referência
    
    for dado in dados_rostos:
        embeddings_conhecidos.append(np.array(dado['embedding']))
        # Assumindo que 'filename' é o campo que referencia a imagem no GridFS
        ids_rostos.append(dado['filename'])

    embeddings_conhecidos = np.array(embeddings_conhecidos)
    
    # 2. Calcula as distâncias (L2 - Euclidean)
    distancias = face_recognition.face_distance(embeddings_conhecidos, novo_embedding)
    
    # 3. Encontra o índice da menor distância (o match)
    indice_mais_proximo = np.argmin(distancias)
    menor_distancia = distancias[indice_mais_proximo]
    
    # 4. Define o limite de similaridade (Threshold)
    LIMITE_SIMILARIDADE = 0.6 
    
    if menor_distancia < LIMITE_SIMILARIDADE:
        # Match satisfatório
        return ids_rostos[indice_mais_proximo], menor_distancia, None
    else:
        # Distância maior que o limite, não é a mesma pessoa.
        return None, menor_distancia, "Face detectada, mas não há um match similar o suficiente na base de dados (distância > 0.6)."


# --- Interface Streamlit Principal ---

st.title("🔎 Atividade: Comparação de Faces com MongoDB e Streamlit")
st.markdown("---")

uploaded_file = st.file_uploader(
    "1. Faça o upload de uma imagem com uma face para comparação:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Exibe a imagem enviada
    st.image(uploaded_file, caption='Imagem enviada', use_container_width=True)
    
    # Converte o arquivo carregado para bytes
    uploaded_bytes = uploaded_file.getvalue()
    
    with st.spinner('2. Processando face e comparando com o banco de dados...'):
        
        # 1. Detectar e codificar a nova face
        novo_embedding, erro_face = detectar_e_codificar_face(uploaded_bytes)
        
        if erro_face:
            st.error(erro_face)
        elif novo_embedding is not None:
            
            # 2. Encontrar a face mais parecida no MongoDB
            filename_match, distancia, erro_match = encontrar_face_mais_parecida(novo_embedding)
            
            st.markdown("---")
            st.subheader("Resultado da Comparação:")

            if erro_match:
                 st.warning(f"Não encontrado um match satisfatório. {erro_match}")
                 st.info(f"A menor distância encontrada foi: **{distancia:.4f}**")
            elif filename_match:
                st.success(f"Match encontrado na base de dados!")
                
                # 3. Recuperar e exibir a imagem do match do GridFS
                try:
                    # Busca a imagem no GridFS pelo filename
                    arquivo_match = fs.find_one({'filename': filename_match})
                    
                    if arquivo_match:
                        dados_match = arquivo_match.read()
                        imagem_match = Image.open(io.BytesIO(dados_match))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                             st.metric("Distância (Similaridade)", f"**{distancia:.4f}**")
                        with col2:
                             st.metric("Arquivo Encontrado", filename_match)

                        st.markdown("**Face Mais Parecida na Base de Dados:**")
                        st.image(imagem_match, caption=f"Match: {filename_match}", use_container_width=True)
                    else:
                        st.error(f"Erro: Imagem '{filename_match}' não encontrada no GridFS.")
                        
                except Exception as e:
                    st.error(f"Erro ao recuperar imagem do GridFS: {e}")
