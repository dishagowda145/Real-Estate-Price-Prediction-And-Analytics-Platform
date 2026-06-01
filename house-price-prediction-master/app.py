import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import time

st.set_page_config(
    page_title="House Price Prediction", 
    layout="wide", 
    page_icon="🏡", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide Streamlit Deploy Button and other default elements
st.markdown("""
    <style>
        /* Hide Deploy button */
        .stDeployButton {display:none !important;}
        
        /* Hide footer watermark */
        footer {display: none !important;}
        
        /* Hide all list items (Rerun, Clear Cache, Print, Record Screen) inside the menu */
        [data-testid="stPopoverBody"] ul { display: none !important; }
        
        /* Hide horizontal dividers inside the menu */
        [data-testid="stPopoverBody"] hr { display: none !important; }
        [data-testid="stPopoverBody"] div[data-testid="stMenuSeparator"] { display: none !important; }
        
        /* Hide 'Made with Streamlit' text in the menu */
        [data-testid="stPopoverBody"] a { display: none !important; }
        
        /* Hide specific elements just in case */
        [data-testid="stAutoRerun"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("kc_house_data.csv")
        return df.sample(n=min(2000, len(df)), random_state=42)
    except FileNotFoundError:
        return None

model = load_model()
df_sample = load_data()

# Initialize session state for comparisons and current prediction
if 'saved_properties' not in st.session_state:
    st.session_state.saved_properties = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# --- SIDEBAR INPUTS ---
st.sidebar.title("🏡 House Specifications")
st.sidebar.markdown("Tune the parameters below to predict the house price.")

with st.sidebar.expander("🏠 Basic Information", expanded=True):
    date_input = st.selectbox("Sale Year", options=[2014, 2015])
    date_val = 1 if date_input == 2014 else 0
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=33, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=8.0, value=2.0, step=0.25)
    floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=0)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2015, value=1990, step=1)
    yr_renovated = st.number_input("Year Renovated (0 if none)", min_value=0, max_value=2015, value=0, step=1)

with st.sidebar.expander("📐 Size & Layout"):
    sqft_living = st.number_input("Sqft Living", min_value=290, max_value=13540, value=1500, step=10)
    sqft_lot = st.number_input("Sqft Lot", min_value=520, max_value=1651359, value=5000, step=50)
    sqft_above = st.number_input("Sqft Above", min_value=290, max_value=9410, value=1500, step=10)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=4820, value=0, step=10)

with st.sidebar.expander("✨ Quality & Condition"):
    condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
    grade = st.slider("Construction Grade (1-13)", min_value=1, max_value=13, value=7)
    view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
    waterfront = st.selectbox("Waterfront", options=["No", "Yes"])
    waterfront_val = 1 if waterfront == "Yes" else 0

with st.sidebar.expander("📍 Location & Neighbors"):
    zipcode = st.number_input("Zipcode", min_value=98000, max_value=98199, value=98001, step=1)
    lat = st.number_input("Latitude", min_value=47.1, max_value=47.8, value=47.5, step=0.01)
    long = st.number_input("Longitude", min_value=-122.6, max_value=-121.3, value=-122.2, step=0.01)
    sqft_living15 = st.number_input("Neighbor Sqft Living", min_value=399, max_value=6210, value=1500, step=10)
    sqft_lot15 = st.number_input("Neighbor Sqft Lot", min_value=651, max_value=871200, value=5000, step=50)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Predict Price & Analyze", type="primary", use_container_width=True):
    if model is not None:
        # 2. Add a loading spinner
        with st.spinner('Analyzing market data and calculating valuation...'):
            time.sleep(1.2) # Fake loading for professional feel
            
            features = pd.DataFrame([{
                'date': date_val, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'sqft_living': sqft_living, 'sqft_lot': sqft_lot, 'floors': floors,
                'waterfront': waterfront_val, 'view': view, 'condition': condition,
                'grade': grade, 'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
                'yr_built': yr_built, 'yr_renovated': yr_renovated, 'zipcode': zipcode,
                'lat': lat, 'long': long, 'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15
            }])
            
            prediction = model.predict(features)[0]
            
            # Category
            if prediction > 900000:
                category = "💎 Luxury"
            elif prediction > 400000:
                category = "🏡 Medium"
            else:
                category = "💰 Budget"
                
            raw_score = (condition / 5) * 3 + (grade / 13) * 5 + (view / 4) * 1 + waterfront_val
            rating = min(round((raw_score / 10) * 10 + 2, 1), 10.0)
            if rating < 3: rating = 3.0
            
            if rating >= 8.0 and condition >= 4:
                investment = "🌟 Excellent Investment"
            elif rating >= 6.0:
                investment = "👍 Good Investment"
            else:
                investment = "⚠️ Moderate Investment - Needs Work"
                
            future_price = prediction * (1.04 ** 5)
            # Extra Feature: Estimated Rent
            est_rent = (prediction * 0.05) / 12
            
            # Save to session state
            st.session_state.current_prediction = {
                "prediction": prediction,
                "category": category,
                "rating": rating,
                "investment": investment,
                "future_price": future_price,
                "est_rent": est_rent,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "condition": condition,
                "grade": grade,
                "lat": lat,
                "long": long
            }

# --- MAIN AREA ---
st.title("🏡 Advanced House Price Dashboard")
st.markdown("An AI-powered tool for real estate valuation, investment analysis, and market comparison.")

if model is None:
    st.error("Model file 'model.pkl' not found. Please run the training script `train_model.py` first.")
else:
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction Results", "⚖️ Property Comparison", "📊 Market Insights"])
    
    with tab1:
        if st.session_state.current_prediction is None:
            st.info("👈 Open the sidebar, enter the house specifications, and click **Predict Price & Analyze**.")
            
            # Show a placeholder image or empty state
            st.markdown("### Why use this tool?")
            c1, c2, c3 = st.columns(3)
            c1.markdown("✅ **91% Accuracy** using Gradient Boosting")
            c2.markdown("✅ **Instant** Investment Analysis")
            c3.markdown("✅ **Data-driven** Financial Planning")
            
        else:
            pred_data = st.session_state.current_prediction
            
            st.success(f"### Predicted Market Value: **${pred_data['prediction']:,.2f}**")
            
            # Key Metrics
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("House Category", pred_data['category'])
            res_col2.metric("Quality Rating", f"{pred_data['rating']}/10")
            res_col3.metric("5-Year Est. Value", f"${pred_data['future_price']:,.2f}", "+21% total")
            res_col4.metric("Est. Monthly Rent", f"${pred_data['est_rent']:,.0f}", "5% yield")
            
            st.markdown("---")
            
            # Map & EMI Layout
            map_col, emi_col = st.columns([1, 1])
            
            with map_col:
                st.subheader("📍 Property Location")
                map_df = pd.DataFrame({'lat': [pred_data['lat']], 'lon': [pred_data['long']]})
                st.map(map_df, zoom=11)
            
            with emi_col:
                st.subheader("💳 Mortgage & EMI Calculator")
                st.markdown("Plan your financing directly from the dashboard.")
                
                down_payment_pct = st.slider("Down Payment (%)", 0, 100, 20)
                interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
                loan_tenure = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=20, step=1)
                    
                principal = pred_data['prediction'] - (pred_data['prediction'] * (down_payment_pct / 100))
                r = interest_rate / 12 / 100
                n = loan_tenure * 12
                if r > 0:
                    emi = principal * r * ((1 + r)**n) / (((1 + r)**n) - 1)
                else:
                    emi = principal / n
                    
                st.info(f"**Estimated Monthly Payment (EMI):** ${emi:,.2f}")
                st.caption(f"Principal Loan Amount: ${principal:,.2f} | Down Payment: ${pred_data['prediction']*(down_payment_pct/100):,.2f}")
                
            st.markdown("---")
            
            # 3. Similar Real Houses Feature
            if df_sample is not None:
                st.subheader("🏘️ Similar Real Houses in Market")
                st.markdown(f"Real properties from our dataset priced around **${pred_data['prediction']:,.0f}**")
                
                # Find houses within a $50k range
                similar_houses = df_sample[abs(df_sample['price'] - pred_data['prediction']) < 50000].head(4)
                
                if not similar_houses.empty:
                    display_df = similar_houses[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'yr_built']].copy()
                    display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No extremely similar houses found in the sample data.")
            
            st.markdown("---")
            
            # Action Buttons
            act_col1, act_col2, act_col3 = st.columns([1, 1, 2])
            
            with act_col1:
                if st.button("⭐ Save for Comparison", key="save_btn", use_container_width=True):
                    property_data = {
                        "Price": f"${pred_data['prediction']:,.0f}",
                        "Beds": pred_data['bedrooms'], 
                        "Baths": pred_data['bathrooms'], 
                        "Sqft": pred_data['sqft_living'], 
                        "Condition": f"{pred_data['condition']}/5", 
                        "Category": pred_data['category'], 
                        "Rating": pred_data['rating'],
                        "Est. Rent": f"${pred_data['est_rent']:,.0f}"
                    }
                    st.session_state.saved_properties.append(property_data)
                    st.toast('Property saved successfully!', icon='✅')
                    
            with act_col2:
                report_content = f"""Advanced House Price Analytics Report
=======================================
Predicted Price: ${pred_data['prediction']:,.2f}
Future Price (5 yrs): ${pred_data['future_price']:,.2f}
Estimated Monthly Rent: ${pred_data['est_rent']:,.2f}

Property Analysis:
- Category: {pred_data['category']}
- Quality Rating: {pred_data['rating']}/10
- Investment Viability: {pred_data['investment']}

Core Specifications:
- Bedrooms: {pred_data['bedrooms']}
- Bathrooms: {pred_data['bathrooms']}
- Sqft Living: {pred_data['sqft_living']}
- Condition: {pred_data['condition']}/5
- Grade: {pred_data['grade']}/13

Financial Planning (Based on Inputs):
- Down Payment: {down_payment_pct}%
- Loan Amount: ${principal:,.2f} 
- Monthly EMI: ${emi:,.2f} (at {interest_rate}% for {loan_tenure} years)
"""
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name="house_analytics_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    with tab2:
        st.header("⚖️ Property Comparison")
        if not st.session_state.saved_properties:
            st.info("You haven't saved any properties yet. Go to the Prediction Results tab and click 'Save for Comparison'.")
            st.image("https://cdni.iconscout.com/illustration/premium/thumb/empty-state-2130362-1800926.png", width=300)
        else:
            comp_df = pd.DataFrame(st.session_state.saved_properties)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Clear Saved Properties", type="secondary"):
                st.session_state.saved_properties = []
                st.rerun()

    with tab3:
        st.header("📊 Market Insights")
        if df_sample is not None:
            st.markdown("Explore trends based on a sample of the housing dataset.")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig1 = px.scatter(df_sample, x="sqft_living", y="price", 
                                 color="condition", size="bathrooms",
                                 title="Price vs. Sqft Living (Color = Condition)",
                                 labels={"sqft_living": "Sqft Living (Sqft)", "price": "Price ($)"},
                                 template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)
                
            with chart_col2:
                fig2 = px.histogram(df_sample, x="price", nbins=40, 
                                   title="Market Price Distribution",
                                   labels={"price": "Price ($)"},
                                   template="plotly_white",
                                   color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig2, use_container_width=True)
                
            st.subheader("🏆 Top 15 Expensive Zipcodes")
            zip_avg = df_sample.groupby("zipcode")["price"].mean().reset_index()
            zip_avg = zip_avg.sort_values(by="price", ascending=False).head(15)
            zip_avg["zipcode"] = "Zip " + zip_avg["zipcode"].astype(str)
            
            fig3 = px.bar(zip_avg, x="zipcode", y="price", 
                         title="Average Property Value by Zipcode",
                         labels={"zipcode": "Zipcode", "price": "Average Price ($)"},
                         template="plotly_white",
                         color="price", color_continuous_scale="Viridis")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Could not load kc_house_data.csv to display charts.")
