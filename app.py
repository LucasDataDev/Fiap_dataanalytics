import streamlit as st
import pandas as pd
import openpyxl as opx
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import MinMax, OneHotEncodingNames, OrdinalFeature

@st.cache_data
def carregar_dados():
    return pd.read_excel("Fiap_dataanalytics/data/obesity_tratada_final.xlsx")

dados = carregar_dados()
st.title('Análise de saúde corporal')
st.write('## Controlar o peso corporal é essencial para prevenir doenças e garantir qualidade de vida. Hábitos saudáveis ajudam a manter o equilíbrio e promovem bem-estar físico e mental.')
st.write('### Vamos iniciar uma importante coleta de dados para analisar o cenário atual do paciente e definir' \
'o grau de sua composição corporal!')

generos = {
    "Homem": 0,
    "Mulher": 1
}

st.write('### Informações pessoais')

st.write("### Genero")
input_genero = st.radio('Selecione o seu gênero:', options=list(generos.keys()))
input_genero = generos.get(input_genero)

st.write("### Idade")
input_idade = float(st.slider('Informe sua idade',18,90))

st.write("### Altura")
input_altura = st.slider('Informe sua altura (m)', min_value=1.0, max_value=2.5, step=0.01)

st.write("### Peso")
input_peso = float(st.slider('Informe seu peso (KG)',30,250))

historico_familiar = {
    "Não": 0,
    "Sim": 1
}

st.write("### Historico Familiar")
input_historico_familiar = st.radio('Você possui histórico de obesidade na familia?', 
                   options=list(historico_familiar.keys()))
input_historico_familiar = historico_familiar.get(input_historico_familiar)

st.write("### Cuidados alimentares")

freq_calorias = {
    "Não": 0,
    "Sim": 1
}

st.write("### Controles de rotina")
input_freq_calorica = st.radio('Você consome alimentos caloricos com frequência?', 
                   options=list(freq_calorias.keys()))
input_freq_calorica = freq_calorias.get(input_freq_calorica)

controle_calorias = {
    "Não": 0,
    "Sim": 1
}

input_controle_calorias = st.radio('Você realiza um controle da quantidade de calorias ingeridas?', 
                   options=list(controle_calorias.keys()))
input_controle_calorias = controle_calorias.get(input_controle_calorias)

fuma = {
    "Não": 0,
    "Sim": 1
}

st.write("### Hábitos")
input_fuma = st.radio('Você tem o hábito de fumar?', options=list(fuma.keys()))
input_fuma = fuma.get(input_fuma)

habito_beber = {
    "Pouco"         : "pouco",
    "Frequentemente": "frequentemente",
    "Sempre"        : "sempre",
    "Não"           : "nao"
}

input_beber = st.selectbox("Você consome bebida alcoolica?",options=list(habito_beber.keys()))
input_beber = habito_beber.get(input_beber)

habito_comer = {
    "Pouco"         : "pouco",
    "Frequentemente": "frequentemente",
    "Sempre"        : "sempre",
    "Não"           : "nao"
}

input_comer = st.selectbox("Você come entre as refeições?",options=list(habito_comer.keys()))
input_comer = habito_comer.get(input_comer)

locomocao = {
    "Transporte publico": "publico",
    "Carro"             : "carro",
    "Moto"              : "moto",
    "Bicicleta"         : "bicicleta",
    "Caminhar"          : "caminhar"
}

st.write("### Forma de locomoção")
input_locomover = st.radio("Qual é sua forma mais comum de locomoção?",options=list(locomocao.keys()))
input_locomover = locomocao.get(input_locomover)

nova_base = [input_genero,
             input_idade,
             input_altura,
             input_peso,
             input_historico_familiar,
             input_freq_calorica,
             input_fuma,
             input_controle_calorias,
             input_comer,
             input_beber,
             input_locomover,
             0
]

def data_split(df, teste_size):
    SEED=42
    treino_df, teste_df = train_test_split(df,test_size=teste_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df,teste_df = data_split(dados, 0.2)

cliente_predict_df = pd.DataFrame([nova_base], columns=teste_df.columns)

teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], ignore_index=True)

def run_pipeline(X):

    pipe = Pipeline([
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMax()),

    ])
    df_pipeline = pipe.fit_transform(X)
    return df_pipeline

teste_novo_cliente = run_pipeline(teste_novo_cliente)

cliente_pred = teste_novo_cliente.drop(['obesidade'],axis=1)

mensagens_resultado = {
    0: "Peso normal",
    1: "Sobrepeso leve",
    2: "Sobrepeso",
    3: "Obesidade Grau I",
    4: "Obesidade Grau II",
    5: "Obesidade Grau III",
    6: "Abaixo do peso"
}

if st.button('Analisar'):
    model = joblib.load('modelo/xbg.joblib')
    final_pred = model.predict(cliente_pred)
    
    resultado = mensagens_resultado[final_pred[-1]]
    st.success(f"### O grau identificado foi: **{resultado}**")

    # Mensagens visuais extras
    if final_pred[-1] >= 3:
        st.error("Atenção: seu grau de obesidade exige cuidados médicos.")
    elif final_pred[-1] == 2:
        st.warning("Fique atento: você está com sobrepeso.")
    elif final_pred[-1] == 1:
        st.info(" Você está com sobrepeso leve.")
    elif final_pred[-1] == 0:
        st.balloons()
        st.success("Parabéns! Você está dentro do peso ideal.")
    elif final_pred[-1] == 6:
        st.warning("Abaixo do peso: mantenha atenção nutricional.")