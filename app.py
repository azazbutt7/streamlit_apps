import streamlit as st
import pickle

model = open('random_forest_8.pkl', 'rb')
rf = pickle.load(model)


def predict_cost(gen_biomass, gen_fb_coal, gen_f_gas, gen_foss_hc,
                 gen_f_oil, gen_hw_reservoir, gen_other, gen_solar):
    pred = rf.predict([[gen_biomass, gen_fb_coal, gen_f_gas, gen_foss_hc,
                        gen_f_oil, gen_hw_reservoir, gen_other, gen_solar]])
    print(pred)
    return pred


def main():
    st.write('''
        # Predicting Cost for Power Consumption
    ''')

    gen_biomass = st.text_input("Generation Biomass", "Type Here")
    gen_fb_coal = st.text_input("Generation Fossil Brown Coal", "Type Here")
    gen_f_gas = st.text_input("Generation Fossil Gas", "Type Here")
    gen_foss_hc = st.text_input("Generation Fossil Hard Coal", "Type Here")
    gen_f_oil = st.text_input("Generation Fossil Oil", "Type Here")
    gen_hw_res = st.text_input("Generation Hydro Water", "Type Here")
    gen_other = st.text_input("Generation Other", "Type Here")
    gen_solar = st.text_input("Generation Solar", "Type Here")

    result = ""

    if st.button("Predict Cost"):
        result = predict_cost(gen_biomass, gen_fb_coal, gen_f_gas, gen_foss_hc,
                              gen_f_oil, gen_hw_res, gen_other, gen_solar)
    st.success("The total estimated cost (in USD) will be {}".format(result))

    if st.button("About"):
        st.text("Lets Learn")
        st.text("Made with Streamlit")


if __name__ == '__main__':
    main()
