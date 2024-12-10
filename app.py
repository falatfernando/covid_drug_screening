
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pycaret.regression import load_model
import plotly.express as px

# Configurações iniciais
st.set_page_config(
    page_title="Predição Bioatividade - COVID-19",
    layout="wide",
    page_icon="🔬"
)

# Carregar o modelo treinado
@st.cache_resource
def load_trained_model():
    return load_model('fine_tuned_model')

model = load_trained_model()

# Carregar colunas reduzidas
@st.cache_resource
def load_reduced_columns():
    reduced_features_df = pd.read_parquet('datasets/reduced_fingerprints.parquet')
    return list(reduced_features_df.columns[:-1])  # Excluir 'pIC50'

reduced_columns = load_reduced_columns()

# Gerador de Morgan Fingerprints
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

# Função para computar Morgan Fingerprints
def compute_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = morgan_gen.GetFingerprint(mol)
        return list(fp)
    else:
        return [0] * 1024

# Sidebar - Informações do App
st.sidebar.title("🔬 Sobre")
st.sidebar.info(
    "Este aplicativo utiliza machine learning para prever a bioatividade de compostos químicos contra a protease 3C-like do SARS-CoV-2.")
st.sidebar.warning("Envie um arquivo CSV contendo moléculas no formato SMILES e baixe ou visualize os resultados diretamente no app.")
st.sidebar.markdown("[🔗 Repositório no GitHub](https://github.com/falatfernando/covid_drug_screening)")

with st.sidebar.expander("ℹ️ Clique aqui para a documentação técnica"):
    st.write("""
    
    O modelo de machine learning aqui presente é um ***Random Forest Regressor*** com **R² de 0.7240 e RMSLE de 0.0802** testado em 5 folds.
                      
    Este repositório foi inspirado no curso de bioinformática com machine learning da FreeCodeCamp.
    Se você deseja replicar o estudo, há um curso completo disponível gratuitamente no YouTube: [Curso no YouTube](https://www.youtube.com/watch?v=jBlTQjcKuaY).
    
    No curso, a proteína do COVID foi substituída por uma acetilcolinesterase devido à falta de informações sobre a proteína original. 
    No entanto, adotei uma abordagem diferente.
    Em vez de trocar minha pipeline para outra biomolécula, permaneci com a ideia original e executei uma análise de Morgan Fingerprints, 
    em vez de uma análise de PubChem Fingerprints. Além de testar técnicas de **data-augmentation** para o dataframe. Duas barreiras encontradas no curso, que induziram a troca pelo professor.
    
    Também aproveitei a experiência para desenvolver com o PyCaret, biblioteca automatizada de machine learning, para obter o melhor modelo possível para essa triagem de medicamentos.
             
    Com carinho,
    Fernando.
    """
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Conecte-se comigo:**")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fernandofalat/)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/falatfernando)")

# Título principal
st.title("🧪 Predição de Bioatividade - COVID-19")
st.markdown("Interações de inibição pIC50 contra a 3C-like protease, ou protease princial Mpro.")
st.markdown("Envie um arquivo CSV com moléculas no formato **SMILES** para prever a bioatividade.")

# Carregar exemplo de arquivo
@st.cache_resource
def load_example_data():
    return pd.DataFrame({
        "SMILES": [
            "CCO",  # Etanol
            "CCCC",  # Butano
            "CC(=O)O",  # Ácido acético
            "CCN(CC)CC",  # Dietilamina
            "CC(C)O"  # Isopropanol
        ]
    })

example_data = load_example_data()

# Opção de usar dados de exemplo
use_example = st.checkbox("Usar dados de exemplo")

if use_example:
    molecules_df = example_data.copy()
    st.success("Dados de exemplo carregados!")
    st.write("Pré-visualização do arquivo:")
    st.dataframe(molecules_df)
else:
    uploaded_file = st.file_uploader("Envie um arquivo CSV contendo uma coluna chamada 'SMILES'", type=["csv"])
    if uploaded_file:
        molecules_df = pd.read_csv(uploaded_file)
        if 'SMILES' not in molecules_df.columns:
            st.error("O arquivo deve conter uma coluna chamada 'SMILES'.")
            molecules_df = None
        else:
            st.success("Arquivo carregado com sucesso!")
            st.write("Pré-visualização do arquivo:")
            st.dataframe(molecules_df.head())
    else:
        molecules_df = None

if molecules_df is not None:
    try:
        # Processar moléculas
        with st.spinner("Processando moléculas..."):
            molecules_df['Fingerprint'] = molecules_df['SMILES'].apply(compute_morgan_fingerprint)

        # Filtrar moléculas válidas
        valid_molecules = molecules_df.dropna(subset=['Fingerprint'])
        if valid_molecules.empty:
            st.error("Nenhuma molécula válida encontrada no arquivo.")
        else:
            st.success("Moléculas processadas com sucesso!")

            # Converter fingerprints para DataFrame
            X = pd.DataFrame(valid_molecules['Fingerprint'].tolist(), columns=[f"morgan_fp_{i}" for i in range(1024)])
            X.columns = [str(i) for i in range(X.shape[1])]
            X_reduced = X[reduced_columns]

            # Realizar predições
            st.write("Realizando predições de bioatividade...")
            predictions = model.predict(X_reduced)
            valid_molecules['Predição_Bioatividade'] = predictions

            # Visualização dos resultados
            st.success("Predições concluídas!")
            st.write("Resultados:")
            st.dataframe(valid_molecules[['SMILES', 'Predição_Bioatividade']])

            # Gráficos interativos
            st.write("Distribuição das predições:")
            fig = px.histogram(valid_molecules, x="Predição_Bioatividade", title="Distribuição de Bioatividade", labels={"Predição_Bioatividade": "Bioatividade"})
            st.plotly_chart(fig, use_container_width=True)

            # Botão para download
            csv = valid_molecules[['SMILES', 'Predição_Bioatividade']].to_csv(index=False)
            st.download_button(
                label="📥 Baixar Predições",
                data=csv,
                file_name='predicoes.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
