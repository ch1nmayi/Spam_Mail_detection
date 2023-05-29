import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Load the saved vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# File path to store counts
COUNTS_FILE = 'counts.txt'

# Load counts from file or initialize if the file is empty
try:
    with open(COUNTS_FILE, 'rb') as f:
        counts = pickle.load(f)
except (IOError, EOFError):
    counts = {'spam_count': [], 'ham_count': []}


def save_counts():
    with open(COUNTS_FILE, 'wb') as f:
        pickle.dump(counts, f)

def clear_file():
    with open(COUNTS_FILE, 'w') as f:
        f.write('')


        
def show_pie_chart():
        labels = ['HAM', 'SPAM']
        sizes = [len(counts['ham_count']), len(counts['spam_count'])]

        colors =["#ff9999" , "#66b3ff"]
         
        if sum(sizes) == 0:
         st.write("##")
         st.error("No data available to display the pie chart.")
         st.write("##")
        else:
         fig, ax = plt.subplots()
         ax.pie(sizes, labels=labels, autopct='%1.1f%%',colors=colors)
         ax.axis('equal')
         plt.title(" ")
         st.pyplot(fig)



def detect_spam(email):
    global counts

    # Transform the input email using the loaded vectorizer
    input_data_features = vectorizer.transform([email])

    
    prediction = loaded_model.predict(input_data_features)

    
    if prediction[0] == 1:
        counts['ham_count'].append(0)
    else:
        counts['spam_count'].append(1)

    save_counts()

    
    if prediction[0] == 1:
        st.header('RESULT: Ham mail')
        
        st.subheader("Total number of spam and ham mail recived :")
        st.write("Ham Count:", len(counts['ham_count']))
        st.write("Spam Count:", len(counts['spam_count']))
    else:
        st.header('RESULT: Spam mail')
       
        st.subheader("Total number of spam and ham mail recived :")
        st.write("Ham Count:", len(counts['ham_count']))
        st.write("Spam Count:", len(counts['spam_count']))

# Streamlit web app
def main():
    st.set_page_config(
        page_title="Spam Mail Detection",
        page_icon="✉",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
   
    #Use local css
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    local_css("style.css")
    
    rad = st.sidebar.radio("NAVIGATION",["📩 Spam Mail Detection"," 👋 About "])
   

    if rad == "📩 Spam Mail Detection":
     st.title('Spam Mail Detection Web App :e-mail: ')
     st.write("---")

     left_column, right_column = st.columns(2)

     with left_column:
        # Get the input email
        email = st.text_area('Enter your email :', height=300)
        st.write("---")
        left_c, right_c = st.columns(2)
        
        # Perform spam detection
        with left_c:
         if st.button('Detect Spam'):
             detect_spam(email)
        with right_c:
            if st.button('Reset'):
             clear_file()
             st.write("Count has been reset.")

     with right_column:
       show_pie_chart()
       
    if rad == " 👋 About ":
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
 
if __name__ == '__main__':
    main()


