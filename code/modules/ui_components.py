import streamlit as st


def inject_custom_css():
    """Inject custom CSS for professional styling"""
    st.markdown(
        """
        <style>
        /* Global Styles */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Card Styling */
        .stMarkdown div {
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Header Styling */
        h1, h2, h3, h4, h5 {
            margin-bottom: 1.5rem !important;
            font-weight: 600 !important;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #2E4057;
            font-size: 1rem;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(46, 64, 87, 0.05);
        }
        
        /* Metric Container Styling */
        .metric-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
        }
        
        /* Button Styling */
        .stButton>button {
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Select Box Styling */
        .stSelectbox [data-baseweb="select"] {
            border-radius: 4px;
        }
        
        /* Slider Styling */
        .stSlider [data-baseweb="slider"] {
            height: 0.3rem;
        }
        
        /* Plot Container Styling */
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Custom Alert Styling */
        .custom-alert {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        
        .custom-alert.info {
            background-color: #E3F2FD;
            border-left: 4px solid #2196F3;
        }
        
        .custom-alert.warning {
            background-color: #FFF3E0;
            border-left: 4px solid #FF9800;
        }
        
        .custom-alert.error {
            background-color: #FFEBEE;
            border-left: 4px solid #F44336;
        }
        
        /* DataTable Styling */
        .dataframe {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe thead tr {
            background-color: #2E4057;
            color: white;
            text-align: left;
        }
        
        .dataframe th,
        .dataframe td {
            padding: 12px 15px;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f8f9fa;
        }
        
        .dataframe tbody tr:last-of-type {
            border-bottom: 2px solid #2E4057;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def create_plot_container(plot_func, title=None):
    """Create a professional container for plots"""
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    if title:
        st.markdown(
            f"<h3 style='margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True
        )
    plot_func()
    st.markdown("</div>", unsafe_allow_html=True)
