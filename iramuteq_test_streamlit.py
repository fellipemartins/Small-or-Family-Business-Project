import streamlit as st
import pandas as pd
import openai
import os
import time
import io
import random
from rapidfuzz import fuzz

# Configure page
st.set_page_config(
    page_title="Iramuteq Abstract Processor",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit Interface
st.title("📚 Iramuteq Abstract Processor - Small or Family Business project")
st.markdown("""
This application helps process academic abstracts by:
- Generating tags based on metadata
- Classifying themes using AI
- Detecting and resolving duplicates
- Generating formatted output files
""")

# Function to get OpenAI API key
def get_openai_api_key():
    """
    Retrieve the OpenAI API key in the following order:
    1. From a local file (apikeys.py)
    2. From Streamlit secrets
    3. From user input
    """
    # Check local file (apikeys.py)
    try:
        import apikeys
        if hasattr(apikeys, "openai"):
            st.info("OpenAI API key loaded from local file (apikeys.py).")
            return apikeys.openai
    except ImportError:
        pass

    # Check Streamlit secrets
    if "openai_api_key" in st.secrets:
        st.info("OpenAI API key loaded from Streamlit secrets.")
        return st.secrets["openai_api_key"]

    # Prompt user for API key
    api_key = st.text_input("Enter your OpenAI API key", type="password", key="openai_api_key")
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to enable theme classification.")
        st.stop()
    return api_key

# Get OpenAI API key
api_key = get_openai_api_key()

# Initialize OpenAI client
try:
    client = openai.OpenAI(
        api_key=api_key,
        timeout=60.0,  # Set timeout to 60 seconds
        max_retries=3  # Set max retries for the client
    )
    # Test the API key with a minimal request
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        st.success("✅ OpenAI API connection successful")
    except openai.AuthenticationError:
        st.error("Authentication failed. Please check your OpenAI API key.")
        st.stop()
    except openai.RateLimitError:
        st.error("Rate limit exceeded. Please try again in a few minutes.")
        st.stop()
    except openai.APIConnectionError as e:
        st.error(f"Unable to connect to OpenAI API. Please check your internet connection. Error: {str(e)}")
        st.stop()
    except Exception as e:
        safe_error = ''.join(c for c in str(e) if ord(c) < 128 and ord(c) >= 32)
        st.error(f"Failed to initialize OpenAI client: {safe_error}")
        st.stop()
except Exception as e:
    safe_error = ''.join(c for c in str(e) if ord(c) < 128 and ord(c) >= 32)
    st.error(f"Failed to create OpenAI client: {safe_error}")
    st.stop()

# Helper function to get unique filenames
def get_unique_filename(base_name, suffix, ext):
    """
    Given a base_name (e.g. "example" or "output_iramuteq"), a suffix (e.g. "_appended" or ""),
    and an extension (e.g. ".xlsx" or ".txt"), return a unique filename.
    If the candidate filename exists, appends a three-digit number (e.g. _002) until a free name is found.
    """
    candidate = f"{base_name}{suffix}{ext}"
    if not os.path.exists(candidate):
        return candidate
    for i in range(2, 1000):
        candidate = f"{base_name}{suffix}{i:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return candidate

# Helper Functions for Tag Generation and Classification
def generate_py_tag(year):
    return f"*py_{year}"

def generate_th_tag(theme):
    theme_norm = theme.strip().lower()
    if "family" in theme_norm:
        return "*th_fam"
    elif "small" in theme_norm:
        return "*th_sma"
    else:
        raise ValueError(f"Unknown theme: {theme}")

def generate_jo_tag(journal):
    words = journal.strip().split()
    letters = ''.join(word[0].lower() for word in words if word)
    return f"*jo_{letters}"

def generate_type_tag(year, theme):
    theme_norm = theme.strip().lower()
    if "family" in theme_norm:
        code = "fam"
    elif "small" in theme_norm:
        code = "sma"
    else:
        raise ValueError(f"Unknown theme: {theme}")
    return f"*type_{code}{year}"

def generate_heading(row):
    py = generate_py_tag(row['publication year'])
    th = generate_th_tag(row['theme'])
    jo = generate_jo_tag(row['journal'])
    type_tag = generate_type_tag(row['publication year'], row['theme'])
    heading = f"{py} {th} {jo} {type_tag}"
    return heading

def classify_theme_openai(abstract):
    prompt = (
        f"Please classify the following abstract as either 'fam' for family business "
        f"or 'sma' for small business. Return only the classification ('fam' or 'sma').\n\nAbstract: {abstract}"
    )
    max_retries = 3
    base_wait_time = 2  # Base wait time in seconds

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0,
                timeout=30.0
            )
            classification = response.choices[0].message.content.strip().lower()
            if classification not in ["fam", "sma"]:
                raise ValueError("Unexpected classification result: " + classification)
            return classification
        except (openai.APIError, openai.APIConnectionError) as e:
            if attempt == max_retries - 1:
                st.error("Failed to connect to OpenAI API after multiple attempts.")
                return "error"
            wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
            st.info(f"Retrying classification in {wait_time:.1f} seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except openai.AuthenticationError:
            st.error("Authentication failed. Please check your OpenAI API key.")
            return "error"
        except openai.RateLimitError:
            st.error("Rate limit exceeded. Please try again in a few minutes.")
            return "error"
        except openai.APIConnectionError:
            st.error("Unable to connect to OpenAI API. Please check your internet connection.")
            return "error"
        except Exception as e:
            safe_error = ''.join(c for c in str(e) if ord(c) < 128 and ord(c) >= 32)
            if not safe_error:
                safe_error = "Unknown error occurred"
            st.error(f"Error: {safe_error}")
            return "error"

def check_consistency(provided_theme, openai_theme):
    provided_norm = provided_theme.strip().lower()
    if "family" in provided_norm:
        expected = "fam"
    elif "small" in provided_norm:
        expected = "sma"
    else:
        raise ValueError(f"Unknown theme: {provided_theme}")
    return "y" if expected == openai_theme else "n"

# File upload with custom styling
st.markdown('<div class="upload-text">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        if not uploaded_file.name.endswith(".xlsx"):
            st.error("Invalid file type. Please upload an Excel file (.xlsx).")
            st.stop()

        with st.spinner("Reading file..."):
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(df.head())

        # Check for required columns
        required_columns = ['paper title', 'publication year', 'theme', 'journal', 'abstract']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()

        # Process data with progress bar
        progress_bar = st.progress(0)
        st.markdown("### 🔄 Processing Data")

        # Process each row: generate tags and get OpenAI classification
        headings = []
        openai_results = []
        consistencies = []
        total_rows = len(df)

        for idx, row in df.iterrows():
            try:
                progress_bar.progress((idx + 1) / total_rows)
                heading = generate_heading(row)
                headings.append(heading)
                classification = classify_theme_openai(row['abstract'])
                openai_results.append(classification)
                consistency = check_consistency(row['theme'], classification)
                consistencies.append(consistency)
            except Exception as e:
                headings.append("error")
                openai_results.append("error")
                consistencies.append("error")
                st.error(f"Error processing row {idx}: {''.join(c for c in str(e) if ord(c) < 128)}")
                break

        df['heading'] = headings
        df['openai'] = openai_results
        df['consistency'] = consistencies

        # Duplicate detection and resolution
        with st.spinner("Detecting duplicates..."):
            df['std_title'] = df['paper title'].str.lower().str.strip()
            df['repeated'] = "n"
            similarity_threshold = 90
            drop_indices = set()
            n = len(df)

            for i in range(n):
                if i in drop_indices:
                    continue
                original_title = df.loc[i, 'std_title']
                original_openai = df.loc[i, 'openai']
                for j in range(i + 1, n):
                    if j in drop_indices:
                        continue
                    duplicate_title = df.loc[j, 'std_title']
                    similarity = fuzz.token_set_ratio(original_title, duplicate_title)
                    if similarity >= similarity_threshold:
                        duplicate_openai = df.loc[j, 'openai']
                        if duplicate_openai == original_openai:
                            drop_indices.add(j)
                        else:
                            if duplicate_openai == "sma":
                                df.at[i, 'theme'] = "small business"
                            elif duplicate_openai == "fam":
                                df.at[i, 'theme'] = "family business"
                            df.at[i, 'openai'] = duplicate_openai
                            df.at[i, 'heading'] = generate_heading(df.loc[i])
                            drop_indices.add(j)
                            df.at[i, 'repeated'] = "y"

        df = df.drop(index=list(drop_indices)).reset_index(drop=True)
        df = df.drop(columns=['std_title'])

        st.markdown("### 📊 Processed Data")
        st.dataframe(df)

        # Generate unique file names for output
        original_filename = uploaded_file.name
        if original_filename.lower().endswith(".xlsx"):
            base_excel = original_filename[:-5]
        else:
            base_excel = original_filename

        appended_excel_filename = get_unique_filename(base_excel, suffix="_appended", ext=".xlsx")
        appended_txt_filename = get_unique_filename("output_iramuteq", suffix="", ext=".txt")

        # Create downloadable outputs
        col1, col2 = st.columns(2)

        with col1:
            # Create Excel output using BytesIO
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)

            excel_data = buffer.getvalue()
            st.download_button(
                "📥 Download Excel Output",
                data=excel_data,
                file_name=appended_excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            txt_lines = ["****"]
            for idx, row in df.iterrows():
                txt_lines.append(row['heading'])
                txt_lines.append(row['abstract'].strip())
                txt_lines.append("****")
            txt_content = "\n".join(txt_lines)
            st.download_button(
                "📥 Download Iramuteq TXT Output",
                data=txt_content,
                file_name=appended_txt_filename,
                mime="text/plain"
            )

        st.success("✅ Processing complete!")

    except Exception as e:
        st.error(f"❌ An error occurred: {''.join(c for c in str(e) if ord(c) < 128)}")