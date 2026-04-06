# Results

## Issues Encountered in Initial Versions:

### The longer we finetuned on VisDrone, the more the model would prefer to detect only VisDrone classes and ignore prompted open-vocab classes

Solutions:

- Three versions were finetuned for varying lengths (80, 40, 5 epochs). The 5 epoch variant showed the most promise in balance between VisDrone classes and Open-Vocab classes.
- During fine-tuning, the CLIP encoder's weights were frozen to prevent them from changing. Only the detection head/neck of YOLO-World architecture were updated to adapt to VisDrone.

### Unable to detect more general obstacles that weren't explicitly prompted for

Solutions:

- Use of descriptive, general prompts to detect "foreign" objects in relevant areas

## Evaluation of Initial Versions