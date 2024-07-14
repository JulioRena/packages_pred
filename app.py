from model_function import *
import streamlit as st
import pandas as pd

loaded_model = load_model()

st.title('Criação de Novo Produto')
st.subheader('Insira os dados do produto:')

altura = st.number_input("Altura")
largura = st.number_input("Largura")
profundidade = st.number_input("Profundidade")
peso = st.number_input("Peso do Produto")
custo = st.number_input("Custo por Unidade de Material")
volume = st.number_input("Volume de Vendas")
material = st.selectbox("Material de Embalagem", ['Papelão', 'Plástico', 'Isopor'])


if st.button('Criar'):
    df = create_df(
        Altura = altura,
        Largura = largura,
        Profundidade = profundidade,
        Peso = peso,
        Custo = custo,
        Volume = volume,
        material = material)
    st.dataframe(df)
    
    df = transform(df, loaded_model=loaded_model)

    text = recommend_dimensions(df,
                                loaded_model= loaded_model, 
                                custo_material = df['Custo por Unidade de Material'].iloc[0])
    
    
    text_format = f'''
    
    
    A Altura recomendada é {text[0].round()}
    
    
    A largura recomendada é {text[1].round()}
            
            
    A profundidade recomendada é {text[2].round()}
    
    
    O volume inicial do produto é: {df['Altura'][0] * df['Largura'][0] * df['Profundidade'][0].round()}
    
    O Volume previsto é de {(text[1] * text[0] * text[2]).round()}
    
    '''
    st.success(text_format, icon="✅")





