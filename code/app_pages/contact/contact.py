import streamlit as st

st.header("Contact Us")
st.markdown("Get in touch for data submission, collaboration, or corrections.")
st.subheader("Submit Your Data")
st.markdown(
    """
    We welcome submissions of new mouse pituitary datasets. To submit your data:
    1. Ensure your raw data is deposited in a public repository (SRA, ENA, GEO, ArrayExpress, etc.)
    2. Fill out our [data submission form](https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUM1o5T01KTkVLQUFKWkFNTU5FVkRBRVoxVy4u&embed=true)
    3. Email us at **epitome at kcl dot ac dot uk** with:
    - Publication details
    - Repository accession numbers
    - Any additional metadata (Genotype, Sex, Age, etc. - see existing curation)
"""
)

st.subheader("Are you sitting on unpublished data?")
st.markdown(
    """
    If you have data that did not make it into a paper, but you would like to share it with the community (and assign a DOI and get cited!), we can help.
    Reach out to us at 
    **epitome at kcl dot ac dot uk**
    with a brief description of your data and we can help you get it into the atlas.
    """
)

st.subheader("Reach Out for Collaboration")
st.markdown(
    """
    We're interested in collaborating on:
    - Including our atlas analysis results in your publication
    - Combining our data with yours to increase statistical power
    - Developing new methods that work on an atlas-scale
    - Adding new modalities (methylation, proteomics, spatial data etc.)
    Contact us at **epitome at kcl dot ac dot uk** with a brief proposal.
"""
)

st.subheader("Work with Us")
st.markdown(
    """
    There is plenty of more work to be done on the systematic analysis of pituitary gland datasets.
    If you are interested in joining our team as a student, please reach out to
    **cynthia dot andoniadou at kcl dot ac dot uk**
    with your CV and a brief statement of interest.
"""
)

st.subheader("Submit a Correction")
st.markdown(
    """
    We are humans too. Did we get something wrong? Did we miss your dataset?
    Please let us know by supplying the relevant information through [this form]("https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUNjlDOURVSTRYMUxHSkpIUDE5OVNUTk1SVS4u&embed=true") or email.

    - Data corrections
    - Metadata updates
    - Website functionality issues

    Email us at: **epitome at kcl dot ac dot uk** with detailed information about the correction needed.
"""
)
