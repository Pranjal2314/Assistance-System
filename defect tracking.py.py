from src.utils.defect_tracker import DefectTracker

tracker = DefectTracker()
defect_id = tracker.log_defect(
    module="NLP Processor",
    description="Incorrect response to weather queries",
    severity="High",
    steps_to_reproduce=["1. Ask 'What's the weather?'", "2. Observe response"]
)