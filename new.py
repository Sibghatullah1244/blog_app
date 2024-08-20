import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_blog(input_text, no_words, blog_style):
  model_name = 'models/adapter_model.bin'  # Replace with your local model path

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

   prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=256, temperature=0.01)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional 2 fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
    try:
        no_words = int(no_words)
    except ValueError:
        st.error("Please enter a valid number for the number of words.")
        no_words = None

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    if input_text and no_words:
        response = generate_blog(input_text, no_words, blog_style)
        st.text_area("Generated Blog", response, height=300)
    else:
        st.error("Please provide valid inputs for all fields.")
