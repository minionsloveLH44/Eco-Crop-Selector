import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
covercrop=pd.read_csv("covercrop.csv")
covercrop["Soil Type"]=covercrop["Soil Type"].astype("category").cat.codes
covercrop["Moisture"]=covercrop["Moisture"].astype("category").cat.codes
covercrop["Erosion Control"]=covercrop["Erosion Control"].astype("category").cat.codes
covercrop["Nitrogen Fixation"]=covercrop["Nitrogen Fixation"].astype("category").cat.codes
covercrop["Weed Suppression"]=covercrop["Weed Suppression"].astype("category").cat.codes
a=covercrop[["pH","Soil Type","Moisture","Temperature","Rainfall","Erosion Control","Nitrogen Fixation","Weed Suppression"]]
b=covercrop["Cover Crop"]
train_a,test_a,train_b,test_b=train_test_split(a,b,test_size=20/100, random_state=1)
scaler1=StandardScaler()
trainsc1=scaler1.fit_transform(train_a)
testsc1=scaler1.transform(test_a)
mdc=RandomForestClassifier(n_estimators=1000, random_state=1)
mdc.fit(trainsc1,train_b)
croprec=pd.read_csv("modified.csv")
ac=croprec[["pH","Temperature","Rainfall"]]
b1=croprec["Soil Type"]
b2=croprec["Recommended crop"]
b3=croprec["Natural Fertilizer"]
train_ac,test_ac,train_b1,test_b1,train_b2,test_b2,train_b3,test_b3=train_test_split(ac,b1,b2,b3,test_size=0.1,random_state=1)
scaler2=StandardScaler()
trainsc2=scaler2.fit_transform(train_ac)
testsc2=scaler2.transform(test_ac)
md1=RandomForestClassifier(n_estimators=1000,random_state=1)
md1.fit(trainsc2,train_b1)
md2=RandomForestClassifier(n_estimators=1000,random_state=1)
md2.fit(trainsc2,train_b2)
md3=RandomForestClassifier(n_estimators=1000,random_state=1)
md3.fit(trainsc2,train_b3)
st.title("ECO CROP SELECTOR")
st.write("Select the function you want to use:")
option=st.selectbox("Choose an option", ["Crop Recommendation", "Cover Crop Selection"])
if option=="Crop Recommendation" :
    st.subheader("Crop Recommendation System")
    pH=st.number_input("Enter pH:", min_value=3.0, max_value=9.5, step=0.1)
    Temperature=st.number_input("Enter temperature(degree C):", min_value=-50, max_value=50)
    Rainfall=st.number_input("Enter Rainfall(mm):", min_value=0, max_value=2000)
    
    if st.button("Predict"):
        ipdata2=pd.DataFrame([[pH,Temperature,Rainfall]],columns=["pH","Temperature","Rainfall"])
        scipdata2=scaler2.transform(ipdata2)
        soil=md1.predict(scipdata2)
        crop=md2.predict(scipdata2)
        fert=md3.predict(scipdata2)
        st.success(f"Soil Type🪴:{soil[0]}")
        st.success(f"Recommended crop🌱:{crop[0]}")
        st.success(f"Natural fertilizer:{fert[0]}")
        st.write(f"### Why {crop[0]} fits Natural Farming 🌱")
        st.write(f"""
        - **Soil Compatibility**: This crop works well with the detected soil type ({soil[0]}).
        - **Water Efficiency**: Known for its water efficiency, making it suitable for natural water conservation practices.
        - **Organic Fertilizer Recommendations**: Compost and green manure can enhance growth without synthetic inputs.
        - **Biodiversity Support**: Encourages diversity in farming systems, beneficial for soil health and pest control.
        - **Tips**:Use clover, legumes, or vetch for natural soil protection 🌱.
               Rotate crops like tomatoes with beans to enrich soil naturally 🌿.
        """)
elif option=="Cover Crop Selection":
    st.subheader("Cover Crop Selection")
    ph=st.number_input("Enter pH", min_value=4.2, max_value=8.0, step=0.1)
    soil=st.selectbox("Enter Soil Type", ["Clay", "Sandy", "Loamy"])
    moisture=st.selectbox("Select the moisture", ["Low", "Moderate", "High"])
    temp=st.number_input("Enter temperature", min_value=0, max_value=50)
    rain=st.number_input("Enter rainfall", min_value=0, max_value=1000)
    erctrl=st.selectbox("Do you want to use it for Erosion Control?", ["Yes", "No"])
    nfix=st.selectbox("Do you want to use it for Nitrogen Fixation?", ["Yes", "No"])
    wassup=st.selectbox("Do you want to use it for Weed Suppression?", ["Yes", "No"])  
    if st.button("Generate Output"):
        soil_code={"Clay":0,"Sandy":1,"Loamy":2}[soil]
        moisture_code={"Low":0,"Moderate":1,"High":2}[moisture]
        erctrl_code=1 if erctrl=="Yes" else 0
        nfix_code=1 if nfix=="Yes" else 0
        wassup_code=1 if wassup=="Yes" else 0

        ipdata1=pd.DataFrame([[
            ph,
            soil_code,
            moisture_code,
            temp,
            rain,
            erctrl_code,
            nfix_code,
            wassup_code
        ]], columns=["pH","Soil Type","Moisture","Temperature","Rainfall","Erosion Control","Nitrogen Fixation","Weed Suppression"])
        
        scipdata1=scaler1.transform(ipdata1)
        cc=mdc.predict(scipdata1)
        st.success(f"Cover Crop:{cc[0]}")
        st.write("🌷🌹💐Close the browser tab to exit the application🌷🌹💐")
