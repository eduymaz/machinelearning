import requests

baseUrl = "http://localhost:3000/products"

def getProducts():
    response = requests.get(baseUrl)
    return response.json()

'''
for product in getProducts():
    print(product.get("name"), "/", product.get("unitPrice"))
'''

def getProductsByCategory(categoryId):
    response = requests.get(baseUrl+"/?categoryId"+str(categoryId))
    return response

'''
for product in getProductsByCategory(5):
    print(product.get("name"), "/", product.get("unitPrice"))
'''

def createProduct(product):
    response = requests.post(baseUrl, json=product)
    return response.json()

'''
productToCreate = {"suppliedId":2, "categoryId":6, "unitPrice":969, "name":"Kalem"}
createProduct(productToCreate)
'''
def updateProduct(id, product):
    response = requests.put(baseUrl+ "/"+str(id), json=product)
    return response.json()

'''
productToUpdate = {"suppliedId":2, "categoryId":6, "unitPrice":969, "name":"Kalem"}
updateProduct("5",productToUpdate)
'''

def updateProductbyPatch(id, product):
    response = requests.patch(baseUrl+ "/"+str(id), json=product)
    return response.json()

'''
productToUpdate = {"suppliedId":2, "categoryId":6, "unitPrice":5555, "name":"Kalem 100"}
updateProductbyPatch("6",productToUpdate)
'''






















#print("Hello world!")
''' 
// object
// array

'''