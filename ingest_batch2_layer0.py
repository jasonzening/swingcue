"""VLM batch2 Layer 0 results — imported by batch2_pipeline.py"""
VLM_BATCH2 = {
    "dtl-ok-1": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at golf address in sunny backyard, iron club, face-on."},
            {"fr":24,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, sunny backyard, face-on."},
            {"fr":48,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Iron swing at impact, face-on."},
            {"fr":73,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Top of backswing, sunny backyard, face-on."},
            {"fr":97,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through from behind, ResinGem shirt."},
        ],
        "verdict":"needs_human","angle":"mixed",
        "reason":"4/5 face-on, 1 DTL (follow-through). Prefix dtl conflicts with predominantly face-on result. Paused per angle-conflict rule.",
    },
    "dtl-ok-2": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at golf address, sunny backyard, face-on."},
            {"fr":70,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through from behind, ResinGems shirt, DTL."},
            {"fr":140,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Iron swing, sunny backyard, face-on."},
            {"fr":210,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Golfer at address, DTL orientation."},
            {"fr":280,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at golf address, predominantly face-on."},
        ],
        "verdict":"needs_human","angle":"mixed",
        "reason":"3 face-on, 2 DTL. Prefix dtl conflicts with mixed VLM result. Paused per angle-conflict rule.",
    },
    "dtl-wrong-1": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":29,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":59,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Iron swing at impact, face-on."},
            {"fr":88,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Iron swing, face-on."},
            {"fr":118,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf swing, face-on."},
        ],
        "verdict":"needs_human","angle":"face-on",
        "reason":"5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle conflict → paused.",
    },
    "dtl-wrong-2": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":33,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":67,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through from behind, DTL."},
            {"fr":100,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":134,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through from behind, DTL."},
        ],
        "verdict":"needs_human","angle":"mixed",
        "reason":"3 face-on, 2 DTL. Prefix dtl-wrong with mixed VLM result → angle conflict → paused.",
    },
    "dtl-wrong-3": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at golf address, face-on."},
            {"fr":83,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":167,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf near impact, face-on."},
            {"fr":251,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":335,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Follow-through, face-on."},
        ],
        "verdict":"needs_human","angle":"face-on",
        "reason":"5/5 face-on. Prefix dtl-wrong expected DTL; VLM consistently face-on → angle conflict → paused.",
    },
    "fo-ok-1": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, gray polo, sunny backyard."},
            {"fr":22,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, iron club, face-on."},
            {"fr":44,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address/impact, yellow practice ball visible."},
            {"fr":67,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Follow-through, face-on."},
            {"fr":89,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Iron swing at impact, face-on."},
        ],
        "verdict":"PASS","angle":"face-on",
        "reason":"5/5 q1=YES,q2=1,q4=YES, consistently face-on. Matches prefix fo-ok.",
    },
    "fo-ok-2": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man executing golf swing at impact, face-on."},
            {"fr":21,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf swing, face-on."},
            {"fr":42,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf swing follow-through, face-on."},
            {"fr":64,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf swing follow-through, face-on."},
            {"fr":85,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Full follow-through with motion blur, face-on."},
        ],
        "verdict":"PASS","angle":"face-on",
        "reason":"5/5 q1=YES,q2=1,q4=YES, consistently face-on. Matches prefix fo-ok.",
    },
    "fo-wrong-1": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, yellow ball, face-on."},
            {"fr":72,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at address over yellow ball, face-on."},
            {"fr":145,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address with iron, face-on."},
            {"fr":218,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at address with yellow ball, face-on."},
            {"fr":291,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man near impact, face-on."},
        ],
        "verdict":"PASS","angle":"face-on",
        "reason":"5/5 pass. Consistently face-on. Note: sampled frames all at address/setup; may contain multiple swings.",
    },
    "fo-wrong-2": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Man at golf address, face-on."},
            {"fr":31,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":63,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address/impact, face-on."},
            {"fr":95,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf swing with alignment sticks visible, face-on."},
            {"fr":127,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":False,"q5_desc":"Golf swing near impact, feet marginally cropped at bottom."},
        ],
        "verdict":"PASS","angle":"face-on",
        "reason":"4/5 hard pass (fr127 feet marginally cropped). Consistently face-on. Meets >=4 threshold.",
    },
    "fo-wrong-3": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":39,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":78,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Impact position, face-on."},
            {"fr":118,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Top of backswing, DTL view."},
            {"fr":157,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through, DTL view."},
        ],
        "verdict":"needs_human","angle":"mixed",
        "reason":"3 face-on, 2 DTL. Prefix fo expected face-on; fr118/fr157 read as DTL → angle conflict → paused.",
    },
    "fo-wrong-4": {
        "frames": [
            {"fr":0,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Young man at golf address, face-on."},
            {"fr":41,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address with iron, face-on."},
            {"fr":83,"q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,"q5_desc":"Golf address, face-on."},
            {"fr":125,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Follow-through DTL view."},
            {"fr":167,"q1_golf":True,"q2_persons":1,"q3_angle":"DTL","q4_fullbody":True,"q5_desc":"Full follow-through DTL, dirt flying at impact."},
        ],
        "verdict":"needs_human","angle":"mixed",
        "reason":"3 face-on, 2 DTL. Prefix fo expected face-on; fr125/fr167 read as DTL → angle conflict → paused.",
    },
}
