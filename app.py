import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_huggingface import HuggingFaceEndpoint

import os
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
## Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # Verify model file exists
        model_path = 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'
        if not os.path.exists(model_path):
            return f"Model file not found: {model_path}"

        ### LLaMA 2 model
        repo_id="mistralai/Mistral-7B-Instruct-v0.3"
        llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token='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')
        
        ## Prompt Template
        template = """
            Write a blog for {blog_style} job profile for a topic {input_text}
            within {no_words} words.
                """
        
        prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                                template=template)
        
        ## Generate the response from the LLaMA 2 model
        # Use invoke instead of __call__
        formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
        response = llm.invoke(formatted_prompt)
        
        return response
    
    except Exception as e:
        return f"Error: {e}"

# Streamlit app setup
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## Create two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    if input_text and no_words:
        with st.spinner("Generating response..."):
            response = getLLamaresponse(input_text, no_words, blog_style)
            if response:
                st.text_area("Generated Blog", response, height=300)
            else:
                st.error("No response generated. Please try again.")
    else:
        st.error("Please provide both a blog topic and the number of words.")
