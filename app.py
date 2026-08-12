Perfect 👍 Nee YOLO classes exactly 2:

overflow → Garbage Overflow Detected → ALERT
normal → No Overflow → No Alert

So previous code lo unna broad keywords avasaram ledu. Exact class matching use cheyyadam best.

Nee app.py lo:

OVERFLOW_KEYWORDS = [
    "overflow",
    "overflowing",
    "garbage overflow",
    "waste overflow",
    "full bin",
    "garbage",
    "waste",
    "litter",
    "full"
]

deenni remove chesi:

OVERFLOW_CLASS = "overflow"
NORMAL_CLASS = "normal"

ani pettu.

Then check_overflow() function ni:

def check_overflow(result):

    if model is None:
        return False

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        class_name = class_name.lower().strip()

        if class_name == OVERFLOW_CLASS:
            return True

    return False
Final behavior
YOLO Detection
      │
      ├── overflow
      │       ↓
      │   🚨 ALERT
      │
      │   Location
      │   Date & Time
      │   Status: Garbage Overflow Detected
      │
      └── normal
              ↓
          No Alert

Important: normal detect ayithe alert chupinchakudadhu. overflow detect ayinappudu maatrame alert message generate avvali.

And idi Camera + Image + Video moodintiki same logic apply avtundi.
