import streamlit as st
import torch
from PIL import Image
import timm
from torchvision import transforms
from openai import OpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
vector_store = FAISS.load_local('./embedding.faiss', embeddings, allow_dangerous_deserialization=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model(
    "resnet50d", pretrained=True, num_classes=12, drop_path_rate=0.05)

model.load_state_dict(torch.load('./model_weights.pth', map_location=device))

model = model.to(device)
model.eval()

data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
data_mean = data_config["mean"]
data_std = data_config["std"]

classes = {0: 'Agaricus', 1: 'Amanita', 2: 'Boletus', 3: 'Cortinarius', 4: 'Entoloma', 5: 'Exidia', 6: 'Hygrocybe', 7: 'Inocybe', 8: 'Lactarius', 9: 'Pluteus', 10: 'Russula', 11: 'Suillus'}

image_size = (256, 256)

transformer = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=data_mean, std=data_std)])

@torch.no_grad()
def classify(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transformer(image)
    image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    index = output.data.numpy().argmax()
    pred = classes[index]
    return pred

client = OpenAI()

def LLM(document, question):
    try:
        content = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""DOCUMENT:
                        {document}

                        QUESTION:
                        {question}

                        INSTRUCTIONS:
                        Answer the users QUESTION using the DOCUMENT text above.
                        Keep your answer ground in the facts of the DOCUMENT.
                        If the DOCUMENT doesn’t contain the facts to answer the QUESTION return NONE.
                        Don't use any keywords about document.
                    """
                }
            ]
        )
        return content.choices[0].message.content
    except Exception as e:
        return "" 
    
def search(query):
    results = vector_store.similarity_search_with_score(query=query, k=3)
    concat_all =  "\n".join(map(lambda doc: doc[0].page_content, results))
    document = "\n".join(line for line in concat_all.splitlines() if line.strip())
    return LLM(document, query)
    

def pick_img():
    uploaded_file = st.file_uploader(
    "Choose a photo", type=["jpg", "jpeg", "png", "gif", "bmp"]
    )
    if uploaded_file:
        st.session_state.img = uploaded_file
        
    picture = st.camera_input("Or take a picture")
    if picture:
        st.session_state.img = picture

if 'img' not in st.session_state:
    st.session_state.img = None
if 'pre_img' not in st.session_state:
    st.session_state.pre_img = None
if 'mushroom_type' not in st.session_state:
    st.session_state.mushroom_type = None

def view():
    st.title("🍄 MassRoom")
    with st.sidebar:
        pick_img()

    if st.session_state.img:
        st.image(st.session_state.img)
        if st.session_state.pre_img != st.session_state.img:
            with st.chat_message("ai", avatar='./avator.jpeg'):
                with st.status("Diagnosing...", expanded=True):
                    st.session_state.mushroom_type = classify(st.session_state.img)
                    st.write(f'Your photo looks like {st.session_state.mushroom_type}')
                    st.write(f'What do you want to know about {st.session_state.mushroom_type}?')
            st.session_state.pre_img = st.session_state.img
        prompt = st.chat_input()
        if prompt:
            with st.chat_message("ai", avatar='./avator.jpeg'):
                with st.status("Diagnosing...", expanded=True):
                    st.write(search(f'I want to know the knowledge about {st.session_state.mushroom_type}:' + prompt))
                    
view()