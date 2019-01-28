# Wave-U-Net Examples

The Wave-U-Net takes raw audio in stereo as input, which is a series of floats between -1 and +1.
The original paper reached very good results, but as I write this, my net explains only around 80 per cent of the target sound. This net clearly needs further training. 

You note in the example that the recovered sound is not as clear as the original. There are also audible traces of the vocals in the instrumental tracks. 

One short example is available, with its five versions:

* The [original_mix_input](https://github.com/anleva/Wave-U-Net/blob/master/examples/original_mix_input.wav?raw=true), which is the audio containing a mix of both vocals and instrumental music. This is the input to the Wave-U-Net. 
* The [original_vocals](https://github.com/anleva/Wave-U-Net/blob/master/examples/original_vocals.wav?raw=true), containing only the original vocal track. 
* The [original_instrumental](https://github.com/anleva/Wave-U-Net/blob/master/examples/original_instrumental.wav?raw=true), containing only the original instrumental track. 
* The [recovered_vocals](https://github.com/anleva/Wave-U-Net/blob/master/examples/recovered_vocals.wav?raw=true), containing the recovered vocal track. This is one of the outputs from the Wave-U-Net, given the original_mix_input. 
* The [recovered_instrumental](https://github.com/anleva/Wave-U-Net/blob/master/examples/recovered_instrumental.wav?raw=true), containing the recovered instrumental track. This is the other output from the Wave-U-Net, given the original_mix_input. 
