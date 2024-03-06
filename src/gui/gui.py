import streamlit as st
import requests
import json

# Title
st.title("Poogle!")

# Expander with instructions
with st.expander("Instrucciones de Uso"):
    st.write("¡Bienvenido a Poogle!")
    st.write(
        "Para buscar, simplemente ingresa tu consulta en la barra de búsqueda y presiona el botón 'Buscar'.")
    st.write("¡Disfruta explorando!")

# Navigation Bar
user_query = st.text_input("Buscar", value="")

# Search Button
if st.button("Buscar"):
    # Call the query function
    # results = buscar_resultados(user_query)
    try:
        response = json.loads(requests.get('http://127.0.0.1:8080/search/' + user_query).text)['results']

        if len(response) == 0:
            st.write("no results")

        for i, doc in enumerate(response):
            st.write(f"{i}: {doc}")

    except:
        st.write("invalid query :'(")
