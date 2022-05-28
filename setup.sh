#mkdir -p ~/.streamlit/
#
#echo "\
#[server]\n\
#port = $PORT\n\
#enableCORS = false\n\
#headless = true\n\
#\n\
#" > ~/.streamlit/config.toml

#mkdir -p ~/.streamlit/
#echo "[theme]
#base='dark'
#primaryColor='#c3baba'
#[server]
#headless = true
#port = $PORT
#enableCORS = false
#" > ~/.streamlit/config.toml

mkdir -p ~/.streamlit/
echo "[theme]
base="dark"
textColor="#e88282"
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml




