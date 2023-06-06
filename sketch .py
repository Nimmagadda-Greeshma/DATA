#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
image=cv2.imread("person.jpg")


# In[3]:


cv2.imshow("person",image)
cv2.waitkey(0)


# In[4]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)


# In[5]:


grey_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image


# In[6]:


plt.imshow(grey_image)


# In[7]:


invert_image=255-grey_image
plt.imshow(invert_image)


# In[10]:


blurred=cv2.GaussianBlur(invert_image,(21,21),0)
plt.imshow(blurred)


# In[11]:


invert_blurred=255-blurred
pencil_sketch=cv2.divide(grey_image,invert_blurred,scale=256.0)
plt.imshow(pencil_sketch)


# In[12]:


cv2.imshow("Photo",pencil_sketch)


# In[13]:


cv2.imshow("Original",image)
cv2.imshow("pencil sketch",pencil_sketch)


# In[ ]:





# In[ ]:




