import torch
from torch.autograd import Variable
from voice_impersonation_utils import *
from voice_impersonation_model import *
import time
import math

basepath = "voice_impersonation_input/"
CONTENT_FILENAME = ""
STYLE_FILENAME = ""
a_content, sr = wav2spectrum(CONTENT_FILENAME)
a_style, sr = wav2spectrum(STYLE_FILENAME)
a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
model = RandomCNN()
model.eval()

a_C_var = Variable(a_content_torch, requires_grad=False).float()
a_S_var = Variable(a_style_torch, requires_grad=False).float()
a_C = model(a_C_var)
a_S = model(a_S_var)

learning_rate = 0.002
a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3, requires_grad=True)
optimizer = torch.optim.Adam([a_G_var])

style_param = 1
content_param = 1e2

num_epochs = 2000
print_every = 100
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    a_G = model(a_G_var)

    content_loss = content_param * compute_content_loss(a_C, a_G)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G)
    loss = content_loss * style_loss
    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print("{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                      epoch / num_epochs * 100,
                                                                                      timeSince(start),
                                                                                      content_loss.item(),
                                                                                      style_loss.item(),
                                                                                      loss.item()))

gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = "new_voice.wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)