# Results

Issues Encountered in Initial Versions:

1. The longer we finetuned on VisDrone, the more the model would prefer to detect only VisDrone classes and ignore prompted open-vocab classes

Solutions:

1. Three versions were finetuned for varying lengths (80, 40, 5 epochs). The 5 epoch variant showed the most promise in balance between VisDrone classes and Open-Vocab classes.
2. During fine-tuning, the CLIP encoder's weights were frozen to prevent them from changing. Only the detection head/neck of YOLO-World architecture were updated to adapt to VisDrone.