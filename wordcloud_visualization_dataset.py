import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Combine all text into a single string
text = ' '.join(df['v2'])

# Generate the word cloud
wordcloud = WordCloud(width=1920, height=1080, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


