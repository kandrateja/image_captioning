from captioning_image import *

class predict_model(tf.keras.Model):

  def __init__(self,encoder,decoder):

    self.encoder = encoder
    self.decoder= decoder
    


  def __call__(self,Image_path):


    result, attention_plot = evaluate(Image_path,self.encoder,self.decoder)
    #After beam search result
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass
    print('Prediction Caption : ', result)
    result = list(result.split(" "))

    plot_attention(Image_path, result, attention_plot)


#loads only one time for encoder/decoder tf lite weight model
final=predict_model('./Attention/saved_models/encoder','./Attention/saved_models/decoder')


#get the prediction
final('/content/drive/MyDrive/Attention/Flickr_Data/Images/2831314869_5025300133.jpg')


dev_images_file = '/content/drive/MyDrive/Attention/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
# Read the validation image names in a set# Read the validation image names in a set
dev_images = set(open(dev_images_file, 'r').read().strip().split('\n'))



scores=[]
count=0
start = time.time()
for element in dev_images:
  count+=1
  
  full_image_path = image_dir+"/"+ element
  image = full_image_path

  row =data[data['filename'] == element] 
  caption=row['caption']
  reference =caption.iloc[1]

  candidate=final(image)
  if(count==10):
    break;

  score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0.5, 0))
  scores.append(score*100)

print ('Time taken to predict images {} sec\n'.format(time.time() - start))
ax = sns.boxplot(x=scores)
ax.set_title('Box blue scores')
ax.set_xlabel('scores range')