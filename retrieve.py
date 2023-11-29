from PIL import Image 
import requests
import os
from clip_retrieval.clip_client import ClipClient, Modality


def retrieve_topk(prompt, k, client):
    phrase = (" ".join(prompt.split(" ")[5:7]).replace(",", "")).replace("/", "")

    exist_0 = os.path.exists("retrieved_images/"+str(phrase)+str(0)+".jpg")
    exist_1 = os.path.exists("retrieved_images/"+str(phrase)+str(1)+".jpg")
    exist_2 = os.path.exists("retrieved_images/"+str(phrase)+str(2)+".jpg")

    if exist_0 and exist_1 and exist_2:
        return
    print(phrase)

    results = client.query(text=prompt)
    images = []

    for result in results:
        if len(images) == k:
            break
        url = result["url"]
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)
        except:
            continue

    if len(images) == 0:
        print(prompt)

    for i, im in enumerate(images):
        im.convert('RGB').save("retrieved_images/"+str(phrase)+str(i)+".jpg")
    
def main():
    client = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-H-14",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=100,
    )

    prompts = []
    with open('prompts.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            prompts.append(x)

    for prompt in prompts:
        retrieve_topk(prompt, 3, client)

if __name__ == "__main__":
    while True:
        try:
            main()
            break # stop the loop if the function completes sucessfully
        
        except KeyboardInterrupt:
            break
        except:
            print("Retrying ... ")