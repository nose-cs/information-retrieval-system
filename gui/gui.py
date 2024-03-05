import streamlit as st

# Title
st.title("Poogle!")
st.markdown("<style>h1{color: blue; text-align: center;}</style>", unsafe_allow_html=True)

# Expander with instructions
with st.expander("Instrucciones de Uso"):
    st.write("¡Bienvenido a Poogle!")
    st.write("Este es un buscador estilo Google. Para buscar, simplemente ingresa tu consulta en la barra de búsqueda y presiona el botón 'Buscar'.")
    st.write("¡Disfruta explorando!")

# Navigation Bar
user_query = st.text_input("Buscar", value="")

# Search Button
if st.button("Buscar"):
    # Call the query function
    # results = buscar_resultados(user_query)

    # Show results
    st.write("Resultados:")
    # for result in results:
    #     st.write(result)
    st.write("Resultado 1")
    st.write("Resultado 2")
    st.write("Resultado 3")
