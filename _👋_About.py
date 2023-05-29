import streamlit as st 
from streamlit_lottie import st_lottie
import requests

st.set_page_config(
    page_title="Spam Mail Detection ",
    page_icon="✉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_lottieurl(url):
     r = requests.get(url)
     if r.status_code !=200:
      return None
     return  r.json()


#Use local css
def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
local_css("style.css")

lottie_ML = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_LmW6VioIWc.json")


with st.container():
  st.title("About")

  st.write("---")
  left_c,right_c = st.columns(2)
  with left_c:
   st.write(" :large_orange_diamond: In this spam email detection model, we use machine learning techniques to automatically identify and separate unwanted spam emails from legitimate ones (called ham emails).")

   st.write(" :large_orange_diamond: We achieve this by training a computer program using a large collection of example emails that are already labeled as spam(0) or ham(1). The program learns patterns and characteristics specific to spam emails and uses this knowledge to classify new, unseen emails.")
 
  with right_c:
    
    st_lottie(lottie_ML, height=300, key="Machine Learning")
 


   

