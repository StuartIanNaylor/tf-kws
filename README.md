# tf-kws
tflite keyword 

AGC used https://github.com/radiocicletta/vlevel
/etc/asound.conf
```
pcm.!default {
    type asym
    playback.pcm "hw:1"
    capture.pcm  "plug:vlevel"
}


ctl.!default {
        type hw
        card 1
}


pcm.vlevel {
    type ladspa
    slave.pcm "plughw:1";
    path "/usr/lib/ladspa/";
    plugins [{
        label vlevel_stereo
        input {
            controls [ 0.5 4 10 ]
        }
    }]
}
```
`python3 kws-stream.py --noise_index=3 --device=13`
