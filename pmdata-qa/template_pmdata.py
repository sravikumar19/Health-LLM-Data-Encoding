from liquid import Template

readiness_question_NL = Template("""Readiness score is an indicator of how prepared our body is for physical activity. It is decided by activity, recent sleep, heart rate variability, and mood.
        The analysis of recent period of {context_length} days sensor readings show:
        Steps: {steps} steps, Burned Calories: {calories} kcal, Resting Heart Rate: {heart_rate} beats/min, Sleep Minutes:
        {sleep} minutes, Mood: {mood} out of 5. In this regard, can you predict a readiness score/level between 0 and 10?""")
        
fatigue_question_NL = Template("""Fatigue monitoring involves tracking signs of tiredness or exhaustion that could impact an individual’s performance or wellbeing. This is often quantified through measures such as changes in heart rate variability and decreased activity levels.
        The analysis of recent period of {context_length} days sensor readings show:
        Steps: {steps} steps, Burned Calories: {calories} kcal, Resting Heart Rate: {heart_Rate} beats/min, Sleep Minutes:
        {sleep} minutes, Mood: {mood} out of 5. In this regard, can you predict a fatigue score of 1.0, 2.0, 3.0, or 4.0?""")

readiness_question_stats = Template("""Readiness score is an indicator of how prepared our body is for physical activity. It is decided by activity, recent sleep, heart rate variability, and mood.
        The analysis of recent period of {context_length} days averaged sensor readings show:
        Steps: {steps} steps, Burned Calories: {calories} kcal, Resting Heart Rate: {heart_rate} beats/min, Sleep Minutes:
        {sleep} minutes, Mood: {mood} out of 5. In this regard, can you predict a readiness score/level between 0 and 10?""")
        
fatigue_question_stats = Template("""Fatigue monitoring involves tracking signs of tiredness or exhaustion that could impact an individual’s performance or wellbeing. This is often quantified through measures such as changes in heart rate variability and decreased activity levels.
        The analysis of recent period of {context_length} days averaged sensor readings show:
        Steps: {steps} steps, Burned Calories: {} kcal, Resting Heart Rate: {heart_rate} beats/min, Sleep Minutes:
        {sleep} minutes, Mood: {mood} out of 5. In this regard, can you predict a fatigue score of 1.0, 2.0, 3.0, or 4.0?""")

readiness_question_visual = Template("""Readiness score is an indicator of how prepared our body is for physical activity. It is decided by activity, recent sleep, heart rate variability, and mood.
        The analysis of recent period of {context_length} days sensor readings is plotted below showing the number steps taken, calories burned (kcal), resting heart rate in beats per minute, and minutes asleep per day. Mood: {mood} out of 5.
        In this regard, predict a readiness score/level between 0 and 10.""")

fatigue_question_visual = Template("""Fatigue monitoring involves tracking signs of tiredness or exhaustion that could impact an individual’s performance or wellbeing. This is often quantified through measures such as changes in heart rate variability and decreased activity levels.
        The analysis of recent period of {context_length} days sensor readings is plotted below showing the number of steps taken, calories burned (kcal), resting heart rate in beats per minute, and minutes asleep per day. Mood: {mood} out of 5.
        In this regard, predict a fatigue score of 1.0, 2.0, 3.0, or 4.0.""")