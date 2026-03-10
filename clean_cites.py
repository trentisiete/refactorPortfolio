import re
import os

file_path = 'src/content/articles/difussion_models.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

citations = {
    'song2021scorebased_sde': 'Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).',
    'songermon2019ncsn': 'Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. In Advances in Neural Information Processing Systems (NeurIPS).',
    'sohl2015deep': 'Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In International Conference on Machine Learning (ICML).',
    'ho2020denoising': 'Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems (NeurIPS).',
    'krizhevsky2009cifar': 'Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.',
    'lecun1998gradient': 'LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.',
    'goodfellow2014generative': 'Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (NeurIPS).',
    'kingma2013auto': 'Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.',
    'dinh2016density': 'Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv preprint arXiv:1605.08803.',
    'kingma2018glow': 'Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. In Advances in neural information processing systems (NeurIPS).',
    'anderson1982reverse': 'Anderson, B. D. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 13(3), 313-326.',
    'vincent2011connection': 'Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7), 1661-1674.',
    'chen2018neuralode': 'Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. In Advances in neural information processing systems (NeurIPS).',
    'hyvarinen2005scorematching': 'Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(Apr), 695-709.',
    'nicholdhariwal2021improved': 'Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning (ICML).',
    'ronneberger2015unet': 'Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (MICCAI).',
    'dhariwal2021diffusion': 'Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS).',
    'zagoruyko2016wide': 'Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. arXiv preprint arXiv:1605.07146.'
}

# My previous script replaced [@key] with [key].
# I'll convert [key] to [^key] for all known citations
for key in citations.keys():
    # It might be in the format [key] or [key; otherkey]
    # Let's just blindly replace `key]` with `^key]` 
    # Actually `[key]` is safer. Let's do `[` immediately followed by key or space
    content = re.sub(r'\[(' + key + r')\]', r'[^\1]', content)
    
    # What if it was [key1; key2] (which my previous regex turned into [key1; key2])
    # Let's replace `; ` with `] [^` to split multiple citations into separate footnotes.
    # We'll just do a pass for that
    pass

# A generic pass: Find text that looks like citation keys (lowercase letters + year + words)
# Our keys all have numbers in them usually `20...` or `19...`
content = re.sub(r'\[([a-z]+(?:20[0-9]{2}|19[0-9]{2})[a-z_]*)\]', r'[^\1]', content)

# To handle [cite1; cite2], let's look for `; ` inside brackets and split them
# This regex is a bit tricky, but basically `[^cite1; cite2]` -> `[^cite1] [^cite2]`
# Or `[cite1; cite2]` -> `[^cite1] [^cite2]`
def split_cites(match):
    inner = match.group(1)
    parts = inner.split('; ')
    return ' '.join([f'[^{p.strip()}]' for p in parts])

# Notice we might have already added `^` to some, so we clean it.
content = re.sub(r'\[\^?([^\]]+;\s+[^\]]+)\]', split_cites, content)

# Finally, append the bibliography map at the bottom of the file
bib_section = '\n\n---\n\n## Referencias Bibliográficas\n\n'
for key, ref in citations.items():
    bib_section += f'[^{key}]: {ref}\n\n'

if '## Referencias Bibliográficas' not in content:
    content += bib_section

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Footnotes conversion complete.")
