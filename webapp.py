import gradio as gr
import pickle
import numpy as np
from Orange.data import * 


def make_prediction(bedrooms1, bathrooms1, stories1, mainroad1,guestroom1,basement1,hotwaterheating1,airconditioning1,parking1,prefarea1,furnishingstatus1):
    
    bedrooms=DiscreteVariable("bedrooms",values=["1","2","3","4","5","6"])
    bathrooms=DiscreteVariable("bathrooms",values=["1","2","3"])
    stories=DiscreteVariable("stories",values=["1","2","3","4"])
    mainroad=DiscreteVariable("mainroad",values=["yes","no"])
    guestroom=DiscreteVariable("guestroom",values=["yes","no"])
    basement=DiscreteVariable("basement",values=["yes","no"])
    hotwaterheating=DiscreteVariable("hotwaterheating",values=["yes","no"])
    airconditioning=DiscreteVariable("airconditioning",values=["yes","no"])
    parking=DiscreteVariable("parking",values=["0","1","2","3"])
    prefarea=DiscreteVariable("prefarea",values=["yes","no"])
    furnishingstatus=DiscreteVariable("furnishingstatus",values=['furnished','semi-furnished','unfurnished'])
    #price=ContinuousVariable("price")
	
    domain=Domain([bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus])
    data=Table(domain,[[bedrooms1, bathrooms1, stories1, mainroad1,guestroom1,basement1,hotwaterheating1,airconditioning1,parking1,prefarea1,furnishingstatus1]])
	
	
    with open("model.pkcls", "rb") as f:
        clf  = pickle.load(f)
        ar=clf(data)
        preds=clf.domain.class_var.str_val(ar)
        #preds = clf.predict_storage(data)
        return preds


#Create the input component for Gradio since we are expecting 10 inputs

NumberOfBedrooms=gr.Dropdown(["1","2","3","4","5","6"],label="How many bedrooms?")
NumberOfBathrooms=gr.Dropdown(["1","2","3"],label="How many bathrooms?")
NumberOfStories=gr.Dropdown(["1","2","3","4"],label="How many stories?")
OnMainRoad=gr.Dropdown(["yes","no"],label="Is the house on a main road?")
HasGuestRoom=gr.Dropdown(["yes","no"],label="Does the house have a guest room?")
HasBasement=gr.Dropdown(["yes","no"],label="Does the house have a basement?")
HasHotWaterHeating=gr.Dropdown(["yes","no"],label="Does the house have hot water heating?")
HasAC=gr.Dropdown(["yes","no"],label="Does the house have air conditioning?")
parkingstatus=gr.Dropdown(["0","1","2","3"],label="How many parking spots?")
PrefArea=gr.Dropdown(["yes","no"],label="Is the house in a preferred area?")
Furnished=gr.Dropdown(['furnished','semi-furnished','unfurnished'],label='Is the house furnished?')



# We create the output
output = gr.Textbox()


app = gr.Interface(fn = make_prediction, inputs=[NumberOfBedrooms, NumberOfBathrooms, NumberOfStories,OnMainRoad,HasGuestRoom,HasBasement,HasHotWaterHeating,HasAC,parkingstatus,PrefArea,Furnished], outputs=output)
app.launch()

git clone https://huggingface.co/spaces/lokeshbollini/propertyprice_prediction