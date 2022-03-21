# Import packages---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
pd.options.display.float_format = '{:.2f}'.format
st.set_option('deprecation.showPyplotGlobalUse', False)

# Input files----------------------------------------------------------------------
products = pd.read_csv('Product_new.csv')
reviews = pd.read_csv('Review_new.zip', lineterminator='\n')
with open('Sim_Results.pkl', 'rb') as f:
    prod_rec = pickle.load(f)
user_rec = pd.read_csv('user_recs.zip')  
#-----------------------------------------------------------------------------------
reviews[['customer_id', 'product_id', 'rating']] = reviews[['customer_id', 'product_id', 'rating']].apply(pd.to_numeric)
# Random products for initial display on tiki
init_display = products.sample(16, replace=False)[['item_id', 'name', 'description', 'price', 'url', 'image']]
#------------------------------------------------------------------------------------
# Search product category in 'name'
def search(str):
    search = [products[products['name']==x] for x in products['name'] if str in x]
    res = pd.concat(search)

    return res.sample(12, replace=False)
# Get product information
def item(id):
  return products.loc[products['item_id']==id]
# Display recommendations
def display_group (lst = init_display):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.container():               
                st.image(lst['image'].tolist()[0])
                st.write(lst['name'].tolist()[0][:50])
                st.write("Price:      " + str(lst['price'].tolist()[0]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[0]))
            with st.container():
                st.image(lst['image'].tolist()[1])
                st.write(lst['name'].tolist()[1][:50])
                st.write("Price:   " + str(lst['price'].tolist()[1]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[1]))
            with st.container():
                st.image(lst['image'].tolist()[2])
                st.write(lst['name'].tolist()[2][:50])
                st.write("Price:   " + str(lst['price'].tolist()[2]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[2]))
        with col2:
            with st.container():
                st.image(lst['image'].tolist()[3])
                st.write(lst['name'].tolist()[3][:50])
                st.write("Price:   " + str(lst['price'].tolist()[3]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[3]))
            with st.container():
                st.image(lst['image'].tolist()[4])
                st.write(lst['name'].tolist()[4][:50])
                st.write("Price:   " + str(lst['price'].tolist()[4]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[4]))
            with st.container():
                st.image(lst['image'].tolist()[5])
                st.write(lst['name'].tolist()[5][:50])
                st.write("Price:   " + str(lst['price'].tolist()[5]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[5]))
        with col3:
            with st.container():
                st.image(lst['image'].tolist()[6])
                st.write(lst['name'].tolist()[6][:50])
                st.write("Price:   " + str(lst['price'].tolist()[6]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[6]))
            with st.container():
                st.image(lst['image'].tolist()[7])
                st.write(lst['name'].tolist()[7][:50])
                st.write("Price:   " + str(lst['price'].tolist()[7]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[7]))
            with st.container():
                st.image(lst['image'].tolist()[8])
                st.write(lst['name'].tolist()[8][:50])
                st.write("Price:   " + str(lst['price'].tolist()[8]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[8]))
        with col4:
            with st.container():
                st.image(lst['image'].tolist()[9])
                st.write(lst['name'].tolist()[9][:50])
                st.write("Price:   " + str(lst['price'].tolist()[9]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[9]))
            with st.container():
                st.image(lst['image'].tolist()[10])
                st.write(lst['name'].tolist()[10][:50])
                st.write("Price:   " + str(lst['price'].tolist()[10]))
                st.write("Product ID: " + str(lst['item_id'].tolist()[10]))
            with st.container():
                st.image(lst['image'].tolist()[11])
                st.write(lst['name'].tolist()[11][:50])
                st.write("Price:   " + str(lst['price'].tolist()[11])) 
                st.write("Product ID: " + str(lst['item_id'].tolist()[11]))  
# Recommended products
def recommend(item_id, num):
    recs = prod_rec[item_id][:num]
    lst = []
    for rec in recs:
        lst.append(int(rec[1]))
    return lst
# --------------------------------------------------------------------------------------
st.title("Data Science Recommender System Project")
st.write('****************************************************************') 
menu = ['Business Objective', 'Build Project', 'Recommender']
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.image('ukraine.jpg')
st.sidebar.image('stand.jpg')
if choice == 'Business Objective':
    st.write(""" 
- Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.
- Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
""")
    st.markdown("##### **OBJECTIVE:**")
    st.write("Xây dựng Recommendation System cho một hoặc một số nhóm hàng \
        hoá trên tiki.vn giúp đề xuất và gợi ý cho khách hàng trong quá trình lựa chọn sản phẩm ")
    st.markdown("##### **ĐỀ XUẤT:**")
    st.write("""
- Content-base filtering algorithms such as Cosine-Similarity or Gensim for new customer product search.
- Collaborative filtering such as ALS or DBSCAN for our regular customers""")
    st.write("In this project, we will employ Cosine-Similarity and ALS to build Tiki Recommender System")

elif choice == 'Build Project':
    st.subheader('Build Project')
    st.markdown('#### 1. Some data')
    st.write('Product')
    st.dataframe(products.head(3))
    st.write('Review')
    st.dataframe(reviews.head(3))
    st.markdown('#### 2. Visualization')
    st.write('##### Giá bán')
    
    fig= plt.figure(figsize=(5, 5))
    products.price.plot(kind='box')
    st.pyplot(fig.figure) 
    
    fig0= plt.figure(figsize=(5, 5))
    products.price.plot(kind='hist', bins=20)
    st.pyplot(fig0.figure) 
    
    st.write('##### Thương hiệu')
    brands = products.groupby('brand')['item_id'].count().sort_values(ascending=False)
    fig1 = brands[1:11].plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Products Items by brand')
    st.pyplot(fig1.figure)
    
    st.write('##### Giá bán theo thương hiệu')
    price_by_brand = products.groupby(by='brand').mean()['price'].sort_values(ascending=False)
    fig2 = price_by_brand[:10].plot(kind='bar')
    plt.ylabel('Price')
    plt.title('Average price by brand')
    st.pyplot(fig2.figure)
    
    st.write('##### Rating')
    fig3= plt.figure(figsize=(5, 5))
    products.rating.plot(kind='hist', bins=100)
    st.pyplot(fig3.figure)
    
    st.write('##### Reviews Distribution')
    fig4= plt.figure(figsize=(5, 5))
    products.rating.plot(kind='density')
    plt.xlim(0,5)
    st.pyplot(fig4.figure)
    
    st.write('##### Average Rating')
    avg_rating_customer = reviews.groupby(by='product_id').mean()['rating'].to_frame().reset_index()
    avg_rating_customer.rename({'rating': 'avg_rating'}, axis=1, inplace=True)
    n_products = products.merge(avg_rating_customer, left_on='item_id', right_on = 'product_id', how='left')
    fig5= plt.figure(figsize=(5, 5))
    n_products['avg_rating'].plot(kind='hist', bins=100)
    st.pyplot(fig5.figure)

    st.write('##### Top 20 products have the most reviews')
    fig6 = plt.figure(figsize = (8, 6))
    top_products = reviews.groupby('product_id').count()['customer_id'].sort_values(ascending=False)[:20]
    top_products.index = products[products.item_id.isin(top_products.index)]['name'].str[:25]
    top_products.plot(kind='bar')
    st.pyplot(fig6.figure)
    
    st.write('##### Top 20 customers do the most reviews')
    top_rating_customers = reviews.groupby('customer_id').count()['product_id'].sort_values(ascending=False)[:20]
    fig7 = plt.figure(figsize=(8,6))
    plt.bar(x=[str(x) for x in top_rating_customers.index], height=top_rating_customers.values)
    plt.xticks(rotation=70)
    st.pyplot(fig7.figure)
    
    st.markdown('#### 3. Build Model')
    st.markdown('##### Cosine-Similarity')
    st.markdown(""" Steps taken:
    -  'underthesea word_tokenize' used to tokenize texts
    -  TfidfVectorizer to number words, eliminate words in vietnamese-stopwords
    -  cosine_similarity to get the matrix of similarity
    -  Based on cosine-similariry matrix, for each product, we can get a certain number of similar products   
             """)
    st.write('The final file, which contains similar products for all products, was built and imported for later \
        building Recommender System')
    st.markdown('##### ALS Model')
    st.write("Due to too much time it takes to tune the model to get the best parameter settings, the model was built \
    in advance.  The recommendation file for all customers was also built and imported.")
    st.write("Combination of Cosine-Similarity and ALS model is used to make product recommendation depending on users are new or old customers.")
    st.markdown('#### 4. Model Evaluation')
    st.write("""
    -  Cosine-Similarity has no method to evaluate the result since it was built based on the similarity among products
    -  The performance of the result can only be judged in real world when it is put in use""")
    st.write('RMSE of the ALS model is ~ 1.05')
    st.write('This model is good enough to build a Recommender System for customers in database')
elif choice == 'Recommender':
    st.markdown("# Hello! Welcome to Tiki Online Electronics")
    st.markdown("###### *** Please clear/reset field before going to the next field - Sorry for the inconvinience***")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.text_input('Customer ID (6177374, 1827148...)')             
    with col2:
        product_id = st.text_input('Product ID (3792857, 1060082...)')            
    with col3:
        option = st.selectbox(
                        'Product Keywords: ',
                        ('Tivi', 'Loa', 'Camera', 'Laptop', 'Tủ lạnh', 'Khác...'),
                        index = 5)
    if customer_id == product_id:
        if option == 'Khác...' :
            st.markdown("##### ***Suggested Products: *** ")
            display_group()
    if customer_id:
        st.markdown("##### ***Suggested Products: *** ")
        recs = user_rec[user_rec['customer_id'] == int(customer_id)].sample(frac=1)     
        lst_id = recs['product_id'].tolist() + [73314682, 48273751]
        cid_display = pd.concat([item(id) for id in lst_id])
        display_group(cid_display)  
    
    if product_id:
        st.write("Your product: ")
        item = item(int(product_id))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(item['image'].values[0])
        with col2:
            with st.container():
                st.write(item['name'].values[0])
                st.write("Price:  " + str(item['price'].values[0]))
                st.write("Product ID:  " + str(item['item_id'].values[0]))
        with col3:
            st.write(item['description'].values[0][:400])      
        st.markdown("##### ***Similar Products: *** ")
        lst = recommend(int(product_id), 10) + [73314682, 48273751]
        cid_display = pd.concat([products[products['item_id'] == x] for x in lst]).sample(frac=1)
        display_group(cid_display)             
    if option != 'Khác...':
        display_group(search(option))

     
        
        
         
        
    