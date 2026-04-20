"""
Sophisticated Multi-Agent LangGraph Workflow for Synthetic Life-Log Generation.

This module generates realistic wearable device recordings that mimic an average person's
daily life, including:
- Multilingual family conversations with language tags [en]/[zh]
- Children interrupting, asking for things, being kids
- Temporal memory (references to previous days, recurring events)
- Weather/seasonal awareness for long-term generation
- Realistic image descriptions in objective third-person

Architecture:
    LifePlanner -> PersonaManager -> [ScenarioDirector -> DialogueWriter -> 
                                      ScenePhotographer -> QualityValidator] (loop)
"""

import os
import csv
import json
import random
import logging
import re
from typing import TypedDict, Literal, List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from groq import Groq

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
WEATHER_PATTERNS = {
    "winter": ["cold and dry", "snowy", "overcast and chilly", "freezing with light snow"],
    "spring": ["mild and pleasant", "light rain", "warming up", "breezy with scattered clouds"],
    "summer": ["hot and humid", "sunny", "thunderstorms in afternoon", "hazy and warm"],
    "fall": ["cool and crisp", "light drizzle", "clear and comfortable", "windy with falling leaves"]
}

LOCATION_BASES = {
    "home": [
        "Home, living room", "Home, kitchen", "Home, bedroom", 
        "Home, dining area", "Home, balcony", "Home, study room"
    ],
    "commute": [
        "Subway Line 2, Shanghai", "Bus 42, heading downtown", 
        "Walking near Jing'an Temple", "Taxi to Pudong"
    ],
    "work": [
        "Office, 23rd floor conference room", "Office, cafeteria",
        "Office, desk area", "Office, break room"
    ],
    "dining": [
        "Paper Stone Bakery, Shanghai", "Family Kitchen Restaurant",
        "Blue Frog near Century Park", "Local noodle shop, Huangpu District"
    ],
    "leisure": [
        "Century Park, Shanghai", "Jing'an Sculpture Park",
        "Shopping mall, Nanjing Road", "Children's playground, Xujiahui"
    ],
    "travel": [
        "Old Town, Dali, Yunnan, China", "Hotel lobby, Chengdu",
        "Train station, Hangzhou", "Airport Terminal 2, Hongqiao"
    ]
}

# --- Persona Definitions ---
FAMILY_PERSONAS = {
    "parent1": {
        "id": 1,
        "role": "Father",
        "age": 38,
        "job": "Tech Startup Founder",
        "personality": "Busy but loving, often distracted by work, tries to be present for family",
        "speech_patterns": ["um", "you know", "so basically", "the thing is"],
        "languages": ["en", "zh"],
        "primary_language": "en"
    },
    "parent2": {
        "id": 2,
        "role": "Mother", 
        "age": 36,
        "job": "University Lecturer",
        "personality": "Organized, caring, often mediates between kids, health-conscious",
        "speech_patterns": ["sweetie", "okay so", "remember when", "how about"],
        "languages": ["en", "zh"],
        "primary_language": "zh"
    },
    "child1": {
        "id": 3,
        "role": "Older Son",
        "age": 10,
        "personality": "Curious, bookworm, sometimes bossy with siblings, loves asking why",
        "speech_patterns": ["but why", "actually", "I know that", "can I"],
        "languages": ["en", "zh"],
        "primary_language": "en"
    },
    "child2": {
        "id": 4,
        "role": "Daughter",
        "age": 7,
        "personality": "Creative, attention-seeking, dramatic, loves playing pretend",
        "speech_patterns": ["mommy look", "I want", "that's not fair", "please please"],
        "languages": ["en", "zh"],
        "primary_language": "en"
    },
    "child3": {
        "id": 5,
        "role": "Younger Son",
        "age": 4,
        "personality": "Energetic, messy, says random things, copies siblings",
        "speech_patterns": ["me too", "no", "why", "look look"],
        "languages": ["en"],  # Youngest mostly speaks English with Chinese words mixed in
        "primary_language": "en"
    }
}

# Recurring characters the family interacts with
RECURRING_CHARACTERS = [
    {"name": "Grandma (Popo)", "relation": "Mother's mother", "speaks": ["zh", "en"]},
    {"name": "Uncle Jason", "relation": "Father's brother", "speaks": ["en"]},
    {"name": "Kara's mom", "relation": "Parent of child 1's classmate", "speaks": ["en"]},
    {"name": "Teacher Liu", "relation": "Child 2's Chinese teacher", "speaks": ["zh"]},
    {"name": "Ayi Wang", "relation": "Part-time housekeeper", "speaks": ["zh"]},
]

# --- Helper Functions ---

def get_model_name():
    return os.environ.get("SYNTH_MODEL", "llama-3.3-70b-versatile")

def get_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat_completion(client, messages, model_name, temperature=0.75, max_tokens=4000):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

def clean_json_response(response_text):
    """Extracts the first valid JSON object by brace counting."""
    # Try markdown code blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match: 
        return match.group(1)
    
    # Brace counting
    start = response_text.find('{')
    if start == -1: 
        return response_text
    
    brace_count = 0
    in_string = False
    escape = False
    
    for i, char in enumerate(response_text[start:], start):
        if char == '"' and not escape:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        escape = char == '\\' and not escape

        if brace_count == 0 and i > start:
            return response_text[start:i+1]
            
    # Fallback
    end = response_text.rfind('}') + 1
    if end > start:
        return response_text[start:end]
    return response_text

def get_season(date: datetime) -> str:
    """Return season based on Northern Hemisphere."""
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

def get_time_slot(hour: int) -> str:
    """Get descriptive time slot."""
    if 6 <= hour < 8:
        return "early_morning"
    elif 8 <= hour < 10:
        return "morning_routine"
    elif 10 <= hour < 12:
        return "late_morning"
    elif 12 <= hour < 14:
        return "lunch"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 19:
        return "evening_commute"
    elif 19 <= hour < 21:
        return "dinner_family"
    elif 21 <= hour < 23:
        return "bedtime"
    else:
        return "sleeping"

# --- State Definition ---

class GraphState(TypedDict):
    # Configuration
    start_date: datetime
    end_date: datetime
    output_csv: str
    
    # Life Context (set once by Life Planner)
    year_context: Optional[Dict]
    persona_profiles: Dict
    recurring_characters: List[Dict]
    
    # Temporal State
    current_time: datetime
    current_weather: str
    current_season: str
    
    # Memory (accumulated)
    memory_log: List[Dict]  # Recent significant events
    conversation_topics: List[str]  # Topics that can be referenced later
    
    # Current Generation
    current_scenario: Optional[Dict]
    present_speakers: Optional[List[int]]  # Speaker IDs present in scene
    current_audio: Optional[str]
    current_image: Optional[str]
    
    # Control
    validation_feedback: Optional[str]
    feedback_source: Optional[str]
    retry_count: int
    generated_count: int

# --- Agent Nodes ---

def life_planner_node(state: GraphState) -> GraphState:
    """
    Runs once at the start. Sets up yearly context including:
    - Major holidays and events
    - Weather patterns by season
    - Whether it's a travel period or normal life
    """
    start = state['start_date']
    end = state['end_date']
    
    # Determine if this is a special period
    duration_days = (end - start).days
    is_holiday_season = start.month in [1, 2, 7, 8, 10, 12]  # CNY, summer, National Day, year-end
    
    # Decide on a scenario flavor
    if duration_days >= 30 and random.random() < 0.3:
        scenario_type = "family_trip"
        base_location = random.choice(["Dali, Yunnan", "Chengdu, Sichuan", "Hangzhou, Zhejiang"])
    elif is_holiday_season and random.random() < 0.5:
        scenario_type = "holiday_at_home"
        base_location = "Shanghai"
    else:
        scenario_type = "normal_life"
        base_location = "Shanghai"
    
    year_context = {
        "scenario_type": scenario_type,
        "base_location": base_location,
        "is_school_break": start.month in [1, 2, 7, 8],
        "upcoming_holidays": [],
        "work_intensity": random.choice(["light", "normal", "busy", "crunch"]),
    }
    
    # Add holidays if they fall in range
    holidays = [
        (1, 1, "New Year"),
        (2, 10, "Chinese New Year"),  # Approximate
        (5, 1, "Labor Day"),
        (6, 1, "Children's Day"),
        (10, 1, "National Day"),
        (12, 25, "Christmas"),
    ]
    for m, d, name in holidays:
        try:
            h_date = datetime(start.year, m, d)
            if start <= h_date <= end:
                year_context["upcoming_holidays"].append({"date": h_date.isoformat(), "name": name})
        except:
            pass
    
    logger.info(f"🗓️ Life Plan: {scenario_type} in {base_location}, work: {year_context['work_intensity']}")
    
    return {
        **state,
        "year_context": year_context,
        "persona_profiles": FAMILY_PERSONAS,
        "recurring_characters": RECURRING_CHARACTERS,
        "memory_log": [],
        "conversation_topics": []
    }


def scenario_director_node(state: GraphState) -> GraphState:
    """
    Decides the next event scenario based on:
    - Time of day
    - Day of week (weekend vs weekday)
    - Current context (travel vs home)
    - Memory of recent events
    """
    current_time = state.get('current_time')
    
    # Initialize or advance time
    if current_time is None:
        current_time = state['start_date'].replace(hour=7, minute=0)
    else:
        if state['retry_count'] == 0:
            # Natural time progression with some randomness
            jump_minutes = random.randint(45, 150)
            current_time += timedelta(minutes=jump_minutes)
    
    # Skip nighttime (23:00 - 06:30)
    if current_time.hour >= 23 or current_time.hour < 7:
        next_morning = current_time.replace(hour=7, minute=random.randint(0, 30), second=0)
        if current_time.hour >= 23:
            next_morning += timedelta(days=1)
        current_time = next_morning
        if state['retry_count'] == 0:
            logger.info(f"💤 Night time... advancing to {current_time}")
    
    # Check if we're past the end date
    if current_time >= state['end_date']:
        return {**state, "current_time": current_time}
    
    # Derive context
    season = get_season(current_time)
    weather = random.choice(WEATHER_PATTERNS[season])
    time_slot = get_time_slot(current_time.hour)
    is_weekend = current_time.weekday() >= 5
    year_ctx = state.get('year_context', {})
    scenario_type = year_ctx.get('scenario_type', 'normal_life')
    
    # Select who is present based on time slot
    if time_slot in ["early_morning", "morning_routine", "bedtime", "dinner_family"]:
        # Full family at home
        present = [1, 2, 3, 4, 5]
    elif time_slot == "lunch" and is_weekend:
        present = [1, 2, 3, 4, 5]
    elif time_slot in ["late_morning", "afternoon"] and not is_weekend:
        # Kids at school, parents might be working
        present = random.choice([[1], [2], [1, 2]])
    elif is_weekend:
        present = [1, 2, 3, 4, 5]
    else:
        present = [1, 2] if random.random() > 0.5 else [1]
    
    # If traveling, family is together
    if scenario_type == "family_trip":
        present = [1, 2, 3, 4, 5]
    
    # Select location
    if scenario_type == "family_trip":
        base = year_ctx.get('base_location', 'Travel destination')
        if time_slot in ["morning_routine", "bedtime"]:
            location = f"Hotel room, {base}"
        elif time_slot == "lunch" or time_slot == "dinner_family":
            location = f"{random.choice(['Local restaurant', 'Street food market', 'Hotel restaurant'])}, {base}"
        else:
            location = f"{random.choice(['Tourist area', 'Temple', 'Park', 'Market'])}, {base}"
    else:
        if time_slot in ["early_morning", "morning_routine", "bedtime"]:
            location = random.choice(LOCATION_BASES["home"])
        elif time_slot == "evening_commute":
            location = random.choice(LOCATION_BASES["commute"])
        elif time_slot in ["late_morning", "afternoon"] and not is_weekend:
            location = random.choice(LOCATION_BASES["work"])
        elif time_slot in ["lunch", "dinner_family"]:
            location = random.choice(LOCATION_BASES["dining"] + LOCATION_BASES["home"])
        else:
            location = random.choice(LOCATION_BASES["leisure"] + LOCATION_BASES["home"])
    
    # Generate activity summary
    client = get_client()
    
    # Get recent memory for context
    recent_memory = state.get('memory_log', [])[-3:]
    memory_str = ""
    if recent_memory:
        memory_str = "Recent events: " + "; ".join([m.get('summary', '') for m in recent_memory])
    
    prompt = f"""Generate a brief, mundane activity description for a family life-log.

Time: {current_time.strftime('%A, %B %d, %Y at %H:%M')}
Location: {location}
Weather: {weather}
Time Slot: {time_slot.replace('_', ' ')}
Present Family Members: {[FAMILY_PERSONAS[f"{'parent' if p <= 2 else 'child'}{p if p <= 2 else p-2}"]["role"] for p in present]}
Context: {"Family vacation trip" if scenario_type == "family_trip" else "Normal daily life"}
{memory_str}

Generate ONE specific, realistic activity (like "ordering breakfast at a cafe", "getting kids ready for school", "afternoon work call").
Must be MUNDANE and REPETITIVE - normal life, not adventures.

Output ONLY JSON:
{{"activity": "Brief activity description", "mood": "calm/busy/playful/tired/stressed"}}"""

    try:
        res = chat_completion(client, [{"role": "user", "content": prompt}], get_model_name())
        scenario = json.loads(clean_json_response(res))
    except Exception as e:
        logger.warning(f"Director fallback due to: {e}")
        scenario = {"activity": f"Family time during {time_slot.replace('_', ' ')}", "mood": "calm"}
    
    scenario['time_str'] = current_time.strftime('%A %d %B %Y, %H:%M')
    scenario['time_formatted'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    scenario['location'] = location
    scenario['is_weekend'] = is_weekend
    scenario['time_slot'] = time_slot
    
    logger.info(f"🎬 [{scenario['time_str']}] {scenario['activity']} @ {location}")
    
    return {
        **state,
        "current_time": current_time,
        "current_weather": weather,
        "current_season": season,
        "current_scenario": scenario,
        "present_speakers": present,
        "validation_feedback": None,
        "feedback_source": None,
        "retry_count": 0
    }


def dialogue_writer_node(state: GraphState) -> GraphState:
    """
    Generates realistic multi-speaker dialogue with:
    - Language tags [en]/[zh]
    - Natural interruptions from kids
    - Personality-appropriate speech patterns
    - References to memory/recent events
    """
    scenario = state['current_scenario']
    present = state['present_speakers']
    personas = state['persona_profiles']
    client = get_client()
    
    # Build persona descriptions
    speaker_info = []
    for sp_id in present:
        key = f"parent{sp_id}" if sp_id <= 2 else f"child{sp_id - 2}"
        p = personas.get(key, {})
        speaker_info.append(
            f"Speaker {sp_id}: {p.get('role', 'Person')}, {p.get('age', 'unknown')} years old. "
            f"Personality: {p.get('personality', '')}. "
            f"Speech patterns: uses {', '.join(p.get('speech_patterns', []))}. "
            f"Languages: {'/'.join(p.get('languages', ['en']))}"
        )
    
    # Get recent conversation topics for continuity
    recent_topics = state.get('conversation_topics', [])[-5:]
    topics_str = ""
    if recent_topics:
        topics_str = f"\nTopics that can be referenced: {', '.join(recent_topics)}"
    
    # Handle retry feedback
    feedback_prompt = ""
    if state.get('validation_feedback') and state.get('feedback_source') == 'dialogue':
        feedback_prompt = f"\n⚠️ PREVIOUS ATTEMPT FAILED: {state['validation_feedback']}\nFix this issue!\n"
    
    prompt = f"""Create a REALISTIC audio transcript from a wearable device recording.

SCENE:
- Time: {scenario['time_str']}
- Location: {scenario['location']}
- Activity: {scenario['activity']}
- Mood: {scenario.get('mood', 'normal')}
- Weather: {state.get('current_weather', 'pleasant')}

SPEAKERS PRESENT:
{chr(10).join(speaker_info)}
{topics_str}

STRICT FORMAT RULES:
1. Use ONLY "Speaker N:" format (e.g., "Speaker 1:", "Speaker 2:") 
2. NO stage directions, NO "(pauses)", NO sound effects
3. Each speaker block is their complete thought before someone else speaks

STYLE REQUIREMENTS:
- VERY natural and raw, like real family conversation
- Kids interrupt randomly about unrelated things (food, toys, "look at this!")
- Adults talk over each other, lose train of thought
- Include filler words: "um", "uh", "so", "like"
- MUNDANE content: food, kids' behavior, scheduling, mild complaints
- If kids present, they MUST speak at least 2-3 times each

LENGTH: 20-40 speaker blocks total.
{feedback_prompt}

Return ONLY the transcript, starting directly with "Speaker N:"."""

    try:
        res = chat_completion(client, [{"role": "user", "content": prompt}], get_model_name(), temperature=0.8)
        # Clean up any accidental prefixes
        audio = res.strip()
        if not audio.startswith("Speaker"):
            # Try to find where Speaker starts
            idx = audio.find("Speaker")
            if idx > 0:
                audio = audio[idx:]
    except Exception as e:
        logger.error(f"Dialogue Writer error: {e}")
        audio = f"Speaker 1:\n\n[en] Hmm, what should we do now?\n\nSpeaker 2:\n\n[en] I'm not sure, let me think."
    
    return {**state, "current_audio": audio}


def scene_photographer_node(state: GraphState) -> GraphState:
    """
    Generates objective, third-person image descriptions.
    No "I", "my", "camera captures", "we see" - just factual descriptions.
    """
    scenario = state['current_scenario']
    client = get_client()
    
    # Handle retry feedback
    feedback_prompt = ""
    if state.get('validation_feedback') and state.get('feedback_source') == 'image':
        feedback_prompt = f"\n⚠️ FIX THIS: {state['validation_feedback']}\n"
    
    prompt = f"""Describe an image captured by a wearable camera in this scene.

Location: {scenario['location']}
Activity: {scenario['activity']}
Time: {scenario['time_str']}
Weather: {state.get('current_weather', 'clear')}

STRICT RULES:
- Start with "A [object]..." or "The [object]..." or "A view of..."
- NEVER use: "I", "my", "me", "we", "camera", "scene shows", "image captures"
- Describe ONLY visible objects, surfaces, lighting
- Be FACTUAL and OBJECTIVE like inventory
- Include specific details: flooring type, furniture style, colors, items on surfaces
- 2-3 sentences maximum

GOOD EXAMPLE:
"A room with light brown wooden flooring in a herringbone pattern. A cream-colored armchair with a blue blanket draped over it sits near a white oval coffee table with a book and a water bottle."

BAD EXAMPLE (don't do this):
"The camera captures a beautiful morning scene. I can see the family eating breakfast."
{feedback_prompt}

Return ONLY the image description."""

    try:
        res = chat_completion(client, [{"role": "user", "content": prompt}], get_model_name(), temperature=0.6)
        image = res.strip()
        # Remove quotes if wrapped
        if image.startswith('"') and image.endswith('"'):
            image = image[1:-1]
    except Exception as e:
        logger.error(f"Scene Photographer error: {e}")
        image = f"A {scenario['location'].lower()} with various furniture and natural lighting from a window."
    
    return {**state, "current_image": image}


def quality_validator_node(state: GraphState) -> GraphState:
    """
    Validates the generated content and saves if valid.
    Returns feedback for retry if invalid.
    """
    audio = state['current_audio']
    image = state['current_image']
    scenario = state['current_scenario']
    
    # --- Audio Validation ---
    # Must have Speaker N: format
    if not re.search(r"Speaker \d+:", audio):
        logger.warning(f"⚠️ Validation failed: Missing 'Speaker N:' format. Retry {state['retry_count'] + 1}/3")
        return {
            **state,
            "validation_feedback": "Audio must use 'Speaker N:' format (e.g., Speaker 1:, Speaker 2:)",
            "feedback_source": "dialogue",
            "retry_count": state['retry_count'] + 1
        }
    
    # Should not have stage directions in dialogue - check ENTIRE audio, not just first 500 chars
    # Expanded pattern to catch more variations
    stage_direction_pattern = r"\((?:pauses?|sighs?|walks?|types?|laughs?|thinking|looks?|smiles?|nods?|shrugs?|gestures?|coughs?|yawns?|interrupts?|continues?|mutters?|whispers?|shouts?)\)"
    if state['retry_count'] >= 1:  # After 1 retry, force it through
        logger.warning(f"⚠️ Max retries reached for stage directions. Forcing save with cleanup.")
        # Aggressively strip out stage directions ourselves
        audio = re.sub(stage_direction_pattern, '', audio, flags=re.IGNORECASE)
        state = {**state, "current_audio": audio}
    elif re.search(stage_direction_pattern, audio, re.IGNORECASE):
        logger.warning(f"⚠️ Validation failed: Stage directions found. Retry {state['retry_count'] + 1}/1")
        return {
            **state,
            "validation_feedback": "CRITICAL: Remove ALL stage directions like (pauses), (sighs), (walks), (laughs), etc. Only spoken words with language tags [en]/[zh]. NO parenthetical actions.",
            "feedback_source": "dialogue",
            "retry_count": state['retry_count'] + 1
        }
    
    # --- Image Validation ---
    # No first person
    if re.search(r"\b(I|my|me|mine)\b", image, re.IGNORECASE):
        logger.warning(f"⚠️ Validation failed: First person in image. Retry {state['retry_count'] + 1}/3")
        return {
            **state,
            "validation_feedback": "Image description used first person ('I', 'my', 'me'). Use only objective third-person description.",
            "feedback_source": "image",
            "retry_count": state['retry_count'] + 1
        }
    
    # No camera references
    if re.search(r"\b(camera|captures?|scene shows|we see|viewer)\b", image, re.IGNORECASE):
        logger.warning(f"⚠️ Validation failed: Camera reference in image. Retry {state['retry_count'] + 1}/3")
        return {
            **state,
            "validation_feedback": "Don't reference the camera or viewer. Just describe visible objects factually.",
            "feedback_source": "image",
            "retry_count": state['retry_count'] + 1
        }
    
    # --- Save Valid Row ---
    new_row = {
        "Time": scenario['time_formatted'],
        "Location": scenario['location'],
        "Audio": audio,
        "Image": image
    }
    
    csv_file = state['output_csv']
    
    try:
        is_empty = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        
        with open(csv_file, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Time", "Location", "Audio", "Image"])
            if is_empty:
                writer.writeheader()
            writer.writerow(new_row)
        logger.info(f"✅ Row {state['generated_count'] + 1} saved.")
    except Exception as e:
        logger.error(f"Failed to save row: {e}")
    
    # Update memory with the activity
    memory_log = state.get('memory_log', [])
    memory_log.append({
        "time": scenario['time_str'],
        "summary": scenario['activity'],
        "location": scenario['location']
    })
    # Keep only last 10 memories
    memory_log = memory_log[-10:]
    
    # Extract conversation topics for future reference
    topics = state.get('conversation_topics', [])
    # Simple topic extraction - look for nouns/activities mentioned
    if "breakfast" in audio.lower() or "lunch" in audio.lower() or "dinner" in audio.lower():
        topics.append("food/meals")
    if "school" in audio.lower() or "homework" in audio.lower():
        topics.append("children's school")
    if "work" in audio.lower() or "meeting" in audio.lower() or "project" in audio.lower():
        topics.append("work projects")
    topics = list(set(topics))[-10:]  # Deduplicate and limit
    
    return {
        **state,
        "generated_count": state['generated_count'] + 1,
        "validation_feedback": None,
        "feedback_source": None,
        "retry_count": 0,
        "current_scenario": None,
        "current_audio": None,
        "current_image": None,
        "memory_log": memory_log,
        "conversation_topics": topics
    }


# --- Routing Logic ---

def router_logic(state: GraphState) -> Literal["life_planner", "scenario_director", "dialogue_writer", "scene_photographer", "quality_validator", "__end__"]:
    """Routes to the next node based on current state."""
    
    # Initial setup
    if state.get('year_context') is None:
        return "life_planner"
    
    # Handle validation failures with retry logic - AGGRESSIVE LIMITS
    if state.get('validation_feedback'):
        # Changed from > 3 to >= 1 (only allow ONE retry)
        if state['retry_count'] >= 1:
            logger.warning(f"🛑 Max retries ({state['retry_count']}) reached. FORCING validation to pass.")
            # Force through validation by clearing feedback
            state = {**state, "validation_feedback": None, "feedback_source": None}
            return "quality_validator"  # Force it to save
        
        source = state.get('feedback_source')
        if source == "dialogue":
            return "dialogue_writer"
        elif source == "image":
            return "scene_photographer"
        return "scenario_director"
    
    # Check if we're done
    current_time = state.get('current_time')
    if current_time and current_time >= state['end_date']:
        return "__end__"
    
    # Normal flow
    if not state.get('current_scenario'):
        return "scenario_director"
    if not state.get('current_audio'):
        return "dialogue_writer"
    if not state.get('current_image'):
        return "scene_photographer"
    
    return "quality_validator"


# --- Main Entry Point ---

def run_synthetic_generation(output_csv: str, start_date_str: str, num_days: int = 7):
    """
    Run the synthetic life-log generation.
    
    Args:
        output_csv: Path to output CSV file
        start_date_str: Start date in YYYY-MM-DD format
        num_days: Number of days to generate (default 7)
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=num_days)
    
    logger.info(f"🚀 Starting Generation: {start_date.date()} to {end_date.date()} -> {output_csv}")
    
    # Build the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("life_planner", life_planner_node)
    workflow.add_node("scenario_director", scenario_director_node)
    workflow.add_node("dialogue_writer", dialogue_writer_node)
    workflow.add_node("scene_photographer", scene_photographer_node)
    workflow.add_node("quality_validator", quality_validator_node)
    
    # Set entry point and add conditional routing
    workflow.set_entry_point("life_planner")
    workflow.add_conditional_edges("life_planner", router_logic)
    workflow.add_conditional_edges("scenario_director", router_logic)
    workflow.add_conditional_edges("dialogue_writer", router_logic)
    workflow.add_conditional_edges("scene_photographer", router_logic)
    workflow.add_conditional_edges("quality_validator", router_logic)
    
    # Compile the workflow
    app = workflow.compile()
    
    # Clear output file if exists
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
            logger.info(f"🗑️ Cleared existing file: {output_csv}")
        except OSError:
            pass
    
    # Initialize state
    initial_state: GraphState = {
        "start_date": start_date,
        "end_date": end_date,
        "output_csv": output_csv,
        "year_context": None,
        "persona_profiles": {},
        "recurring_characters": [],
        "current_time": None,
        "current_weather": "",
        "current_season": "",
        "memory_log": [],
        "conversation_topics": [],
        "current_scenario": None,
        "present_speakers": None,
        "current_audio": None,
        "current_image": None,
        "validation_feedback": None,
        "feedback_source": None,
        "retry_count": 0,
        "generated_count": 0
    }
    
    try:
        # Calculate a reasonable recursion limit
        # Approximately 4-8 entries per day, ~6 steps per entry + some retries
        limit = max(500, num_days * 10 * 8)
        
        final_state = app.invoke(initial_state, {"recursion_limit": limit})
        
        logger.info(f"🎉 Generation Complete! Total entries: {final_state['generated_count']}")
        logger.info(f"📄 Output saved to: {output_csv}")
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_synthetic_generation(
        output_csv="/workspaces/graphknows/input/synthetic_test_data.csv",
        start_date_str="2025-12-20",
        num_days=5  # Generate 5 days by default for testing
    )
