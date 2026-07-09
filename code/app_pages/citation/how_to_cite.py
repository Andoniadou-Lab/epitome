import streamlit as st

from modules.citations import epitome_citation, print_citation

st.header("How to Cite")
st.markdown("Guide on citing the epitome and original datasets.")

st.subheader("Cite Us")

st.markdown("##### Citing the Consensus Pituitary Atlas")
st.markdown(
    f"""
    When referring to results or methods from the atlas, please cite our Cell Reports publication:

    {print_citation}
"""
)

st.markdown("##### Citing the Epitome")
st.markdown(
    f"""
    When using the website to access data, generate hypotheses, or create figures, please cite:

    {epitome_citation}
"""
)

st.markdown("---")

st.markdown("##### Examples")
st.markdown(
    f"""
    Scenario 1: You have used a result from our Consensus Pituitary Atlas publication, but not the epitome.

    "Gal is more abundant in female mouse pituitaries compared to male ones [1]."

    1. {print_citation}

    Scenario 2: You are retrieving a uniformly pre-processed dataset.

    "To evaluate whether our gene of interest, Bean1, is affected by Prop1, we retrieved a Prop1 knockout dataset [1] from the electronic pituitary omics platform [2].

    1. Cite source paper of dataset 
    2. {epitome_citation}

    Scenario 3: You have used the Epitome to access the atlas, and then created a figure.

    "Using the electronic pituitary omics platform [1] which collates all existing single-cell transcriptomic data on the pituitary [2], we found that our gene of interest, Bean1, is mostly present in gonadotrophs ."

    1. {epitome_citation}
    2. {print_citation}


"""
)
st.markdown("---")

st.subheader("Cite Others")
st.markdown(
    """
    Please also cite the relevant original publications, when you use a single or few datasets:

    1.  Ruf-Zamojski F, Zhang Z, Zamojski M, Smith GR, Mendelev N, Liu H, et al. Single nucleus multi-omics regulatory landscape of the murine pituitary. Nat Commun. 2021 May 11;12(1):2677.
    2.  Cheung LYM, George AS, McGee SR, Daly AZ, Brinkmeier ML, Ellsworth BS, et al. Single-Cell RNA Sequencing Reveals Novel Markers of Male Pituitary Stem Cells and Hormone-Producing Cell Types. Endocrinology. 2018 Dec 1;159(12):3910–24.
    3.	Mayran A, Sochodolsky K, Khetchoumian K, Harris J, Gauthier Y, Bemmo A, et al. Pioneer and nonpioneer factor cooperation drives lineage specific chromatin opening. Nat Commun. 2019 Aug 23;10(1):3807.
    4.	Chen Q, Leshkowitz D, Blechman J, Levkowitz G. Single-Cell Molecular and Cellular Architecture of the Mouse Neurohypophysis. eNeuro. 2020;7(1):ENEURO.0345-19.2019.
    5.	Ho Y, Hu P, Peel MT, Chen S, Camara PG, Epstein DJ, et al. Single-cell transcriptomic analysis of adult mouse pituitary reveals sexual dimorphism and physiologic demand-induced cellular plasticity. Protein Cell. 2020 Aug;11(8):565–83.
    6.	Lopez JP, Brivio E, Santambrogio A, De Donno C, Kos A, Peters M, et al. Single-cell molecular profiling of all three components of the HPA axis reveals adrenal ABCB1 as a regulator of stress adaptation. Sci Adv. 2021 Jan;7(5):eabe4497.
    7.	Ruggiero-Ruff RE, Le BH, Villa PA, Lainez NM, Athul SW, Das P, et al. Single-Cell Transcriptomics Identifies Pituitary Gland Changes in Diet-Induced Obesity in Male Mice. Endocrinology. 2024 Mar 1;165(3):bqad196.
    8.	Vennekens A, Laporte E, Hermans F, Cox B, Modave E, Janiszewski A, et al. Interleukin-6 is an activator of pituitary stem cells upon local damage, a competence quenched in the aging gland. Proc Natl Acad Sci U S A. 2021 Jun 22;118(25):e2100052118.
    9.	Laporte E, Hermans F, De Vriendt S, Vennekens A, Lambrechts D, Nys C, et al. Decoding the activated stem cell phenotype of the neonatally maturing pituitary. eLife. 2022 Jun 14;11:e75742.
    10.	Li Y, Wang J, Wang R, Chang Y, Wang X. Gut bacteria induce IgA expression in pituitary hormone-secreting cells during aging. iScience. 2023 Oct 20;26(10):107747.
    11.	Miles TK, Odle AK, Byrum SD, Lagasse A, Haney A, Ortega VG, et al. Anterior Pituitary Transcriptomics Following a High-Fat Diet: Impact of Oxidative Stress on Cell Metabolism. Endocrinology. 2023 Dec 23;165(2):bqad191.
    12.	Bohaczuk SC, Thackray VG, Shen J, Skowronska-Krawczyk D, Mellon PL. FSHB  Transcription is Regulated by a Novel 5′ Distal Enhancer With a Fertility-Associated Single Nucleotide Polymorphism. Endocrinology. 2021 Jan 1;162(1):bqaa181.
    13.	Schang G, Ongaro L, Brûlé E, Zhou X, Wang Y, Boehm U, et al. Transcription factor GATA2 may potentiate follicle-stimulating hormone production in mice via induction of the BMP antagonist gremlin in gonadotrope cells. J Biol Chem. 2022 Jul 1;298(7):102072.
    14.	Lin YF, Schang G, Buddle ERS, Schultz H, Willis TL, Ruf-Zamojski F, et al. Steroidogenic Factor 1 Regulates Transcription of the Inhibin B Coreceptor in Pituitary Gonadotrope Cells. Endocrinology. 2022 Aug 12;163(11):bqac131.
    15.	Rizzoti K, Chakravarty P, Sheridan D, Lovell-Badge R. SOX9-positive pituitary stem cells differ according to their position in the gland and maintenance of their progeny depends on context. Sci Adv. 9(40):eadf6911.
    16.	Cheung LYM, Menage L, Rizzoti K, Hamilton G, Dumontet T, Basham K, et al. Novel Candidate Regulators and Developmental Trajectory of Pituitary Thyrotropes. Endocrinology. 2023 May 15;164(6):bqad076.
    17.	Allensworth-James M, Banik J, Odle A, Hardy L, Lagasse A, Moreira ARS, et al. Control of the Anterior Pituitary Cell Lineage Regulator POU1F1 by the Stem Cell Determinant Musashi. Endocrinology. 2021 Mar 1;162(3):bqaa245.
    18.	Moncho-Amor V, Chakravarty P, Galichet C, Matheu A, Lovell-Badge R, Rizzoti K. SOX2 is required independently in both stem and differentiated cells for pituitary tumorigenesis in p27-null mice. Proc Natl Acad Sci U S A. 2021 Feb 16;118(7):e2017115118.
    19.	Bastedo WE, Scott RW, Arostegui M, Underhill TM. Single-cell analysis of mesenchymal cells in permeable neural vasculature reveals novel diverse subpopulations of fibroblasts. Fluids Barriers CNS. 2024 Apr 5;21(1):31.
    20.	Cheung LYM, Camper SA. PROP1-Dependent Retinoic Acid Signaling Regulates Developmental Pituitary Morphogenesis and Hormone Expression. Endocrinology. 2020 Jan 8;161(2):bqaa002.
    21.	Zhang Z, Ruf-Zamojski F, Zamojski M, Bernard DJ, Chen X, Troyanskaya OG, et al. Peak-agnostic high-resolution cis-regulatory circuitry mapping using single cell multiome data. Nucleic Acids Res. 2024 Jan 25;52(2):572–82.
    22.	Masser BE, Brinkmeier ML, Lin Y, Liu Q, Miyazaki A, Nayeem J, et al. Gene Misexpression in a Smoc2+ve/Sox2-Low Population in Juvenile Prop1-Mutant Pituitary Gland. J Endocr Soc. 2024 Oct 1;8(10):bvae146.
    23.	Sheridan D, Chakravarty P, Golan G, Shiakola Y, Galichet C, Mollard P, et al. Gonadotrophs have a dual origin, with most derived from pituitary stem cells during minipuberty [Internet]. bioRxiv; 2024 [cited 2024 Sep 13]. p. 2024.09.09.610834. Available from: https://www.biorxiv.org/content/10.1101/2024.09.09.610834v2
    24.	Qian Q, Li M, Zhang Z, Davis SW, Rahmouni K, Norris AW, et al. Obesity disrupts the pituitary-hepatic UPR communication leading to NAFLD progression. Cell Metab. 2024 Jul 2;36(7):1550-1565.e9.
    25.	Martinez-Mayer J, Brinkmeier ML, O’Connell SP, Ukagwu A, Marti MA, Miras M, et al. Knockout mice with pituitary malformations help identify human cases of hypopituitarism. Genome Med. 2024 May 31;16:75.
    26.	Kim Y bin, Lee S, Kim NY, Lee EJ, Oh JH, Oh CM, et al. 12404 Aging-Associated Decline in the Pituitary Gland Using Single-Cell Transcriptomes. J Endocr Soc. 2024 Oct 1;8(Supplement_1):bvae163.1127.
    27.	Khetchoumian K, Sochodolsky K, Lafont C, Gouhier A, Bemmo A, Kherdjemil Y, et al. Paracrine FGF1 signaling directs pituitary architecture and size. Proc Natl Acad Sci. 2024 Oct;121(40):e2410269121.
    28.	Wang Y, Thistlethwaite W, Tadych A, Ruf-Zamojski F, Bernard DJ, Cappuccio A, et al. Automated single-cell omics end-to-end framework with data-driven batch inference. Cell Syst. 2024 Oct 16;15(10):982-990.e5.
    29.	Huang Y, Wang Q, Zhou W, Jiang Y, He K, Huang W, et al. Prenatal p25-activated Cdk5 induces pituitary tumorigenesis through MCM2 phosphorylation-mediated cell proliferation. Neoplasia N Y N. 2024 Oct 3;57:101054.
    30.	Vriendt SD, Laporte E, Abaylı B, Hoekx J, Hermans F, Lambrechts D, et al. Single-cell transcriptome atlas of male mouse pituitary across postnatal life highlighting its stem cell landscape. iScience [Internet]. 2025 Feb 21 [cited 2025 Mar 13];28(2). Available from: https://www.cell.com/iscience/abstract/S2589-0042(24)02935-3
    31.	Brinkmeier ML, Cheung LYM, O’Connell SP, Gutierrez DK, Rhoads EC, Camper SA, et al. Nucleoredoxin regulates WNT signaling during pituitary stem cell differentiation. Hum Mol Genet. 2025 Mar 5;ddaf032.
    32.	Ongaro L, Zhou X, Wang Y, Schultz H, Zhou Z, Buddle ERS, et al. Muscle-derived myostatin is a major endocrine driver of follicle-stimulating hormone synthesis. Science. 2025 Jan 17;387(6731):329–36.
    33.	Miles TK, Odle AK, Byrum SD, Lagasse AN, Haney AC, Ortega VG, et al. Ablation of Leptin Receptor Signaling Alters Somatotrope Transcriptome Maturation in Female Mice. Endocrinology. 2025 Feb 18;bqaf036.
    34. Rebboah, E., Weber, R., Abdollahzadeh, E., Swarna, N., Sullivan, D.K., Trout, D., Reese, F., Liang, H.Y., Filimban, G., Mahdipoor, P., et al. (2025). Systematic cell-type resolved transcriptomes of 8 tissues in 8 lab and wild-derived mouse strains capture global and local expression variation. Cell Genomics, 101108. https://doi.org/10.1016/j.xgen.2025.101108. 
    35.	Cheung, L. (2025). Fundamental mechanisms causing pituitary stem cell aging in mice and humans. (Gene Expression Omnibus). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299835 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299835. 
    36. Wei, R., Du, Z., Tao, W., and Zhang, C. (2025). Single-cell transcriptomic analysis reveals that inflammation drives the unfolded protein response during endocrine aging in mice. (Gene Expression Omnibus). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE239316 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE239316. 
    37. Jin, Y., Schultz, H., Ongaro, L., Schang, G., Zhou, X., Alonso, C.A.I., Zamojski, M., Nudelman, G., Mendelev, N., Onuma, S., et al. (2026). Regulation of murine follicle-stimulating hormone β subunit transcription by newly identified enhancers. Endocrinology, bqag020. https://doi.org/10.1210/endocr/bqag020. 
    38.	Guo, H. (2025). Single-cell analysis results of pituitary tissue from normal diet (ND) mice and high-fat diet (HFD) mice. (Gene Expression Omnibus). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE310493 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE310493. 
    39.	Sochodolsky, K. (2026). BDNF engages pituitary stem cells for establishment of the adult gland and for homeostasis of the corticotrope lineage. (Gene Expression Omnibus). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE316726 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE316726. 
   """
)
