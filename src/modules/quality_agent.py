'''import ollama
import json
import numpy as np
from typing import List, Dict, Any
import logging
import time
import re

logger = logging.getLogger(__name__)

class QualityCheckAgent:

    def __init__(self, model_name="llama3.1:latest"):
        """
        AI Quality Check Agent using Ollama
        Evaluates diarization output across multiple dimensions
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Add retry parameters
        self.max_retries = 3
        self.retry_delay = 1.0

        try:
            # `list()` might return a dict or a list depending on Ollama version
            models_response = self.client.list()

            # Handle various formats
            if hasattr(models_response, 'models'):
                model_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
            elif isinstance(models_response, list):
                model_list = models_response
            else:
                logger.warning(f"Unexpected model list structure: {models_response}")
                model_list = []

            # Extract model names
            available_models = []
            for m in model_list:
                try:
                    model_name_entry = getattr(m, 'name', None)
                    if model_name_entry is None and isinstance(m, dict):
                        model_name_entry = m.get('name')
                    if model_name_entry:
                        available_models.append(model_name_entry)
                except Exception as inner_e:
                    logger.warning(f"Failed to parse model entry: {m} - {inner_e}")

            # Validate model name
            if self.model_name not in available_models:
                logger.warning(f"Model '{self.model_name}' not found. Available: {available_models}")
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"Falling back to available model: {self.model_name}")
                else:
                    logger.warning("No available models found. Attempting to pull the requested model...")
                    try:
                        self.client.pull(self.model_name)
                        logger.info(f"Successfully pulled model: {self.model_name}")
                    except Exception as pull_e:
                        logger.error(f"Failed to pull model {self.model_name}: {pull_e}")
                        raise RuntimeError("No available models found and failed to pull requested model.")

            logger.info(f"Quality check agent initialized with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

    def _make_llm_request(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Make a request to the LLM with retry logic and better error handling
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={
                        'temperature': 0.1,
                        'num_predict': max_tokens
                    }
                )
                
                if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
                    return response['message']['content']
                else:
                    logger.warning(f"Unexpected response format: {response}")
                    return str(response)
                    
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e

    def _extract_json_from_response(self, content: str) -> dict:
        """
        Extract JSON from LLM response with better error handling
        """
        # Try different extraction methods
        extraction_methods = [
            # Method 1: Look for ```json blocks
            lambda x: re.search(r'```json\s*(\{.*?\})\s*```', x, re.DOTALL),
            # Method 2: Look for ``` blocks
            lambda x: re.search(r'```\s*(\{.*?\})\s*```', x, re.DOTALL),
            # Method 3: Look for any JSON object
            lambda x: re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', x, re.DOTALL),
            # Method 4: Entire content if it looks like JSON
            lambda x: re.match(r'^\s*\{.*\}\s*$', x, re.DOTALL)
        ]
        
        for method in extraction_methods:
            match = method(content)
            if match:
                try:
                    json_str = match.group(1) if hasattr(match, 'group') else content
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If all methods fail, return a default structure
        logger.warning(f"Could not extract JSON from response: {content[:200]}...")
        return {}

    def evaluate_segment_quality(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate quality of a single segment
        """
        try:
            prompt = f"""
            Analyze this speaker diarization segment for quality:
            
            Speaker: {segment['speaker']}
            Start: {segment['start']:.2f}s
            End: {segment['end']:.2f}s  
            Duration: {segment['end'] - segment['start']:.2f}s
            Text: "{segment['text']}"
            Confidence: {segment.get('confidence', 0.0):.2f}
            Word Count: {segment.get('word_count', 0)}
            
            Evaluate on these criteria (score 0-10):
            1. Text Completeness: Does the text seem complete/coherent?
            2. Duration Appropriateness: Is the duration reasonable for the text?
            3. Speaker Consistency: Does the text sound like it's from one speaker?
            4. Transcription Quality: Are there obvious transcription errors?
            
            Respond ONLY in this JSON format (no other text):
            {{
                "text_completeness": 8.5,
                "duration_appropriateness": 7.0,
                "speaker_consistency": 9.0,
                "transcription_quality": 8.0,
                "overall_confidence": 8.1,
                "issues": ["specific issue 1", "specific issue 2"],
                "feedback": "Brief qualitative feedback"
            }}
            """
            
            content = self._make_llm_request(prompt, max_tokens=500)
            result = self._extract_json_from_response(content)
            
            # Ensure all required fields exist with defaults
            default_result = {
                "text_completeness": 5.0,
                "duration_appropriateness": 5.0,
                "speaker_consistency": 5.0,
                "transcription_quality": 5.0,
                "overall_confidence": 5.0,
                "issues": [],
                "feedback": "Unable to analyze segment properly"
            }
            
            # Update with extracted values
            for key, value in result.items():
                if key in default_result:
                    default_result[key] = value
            
            # Add segment metadata
            default_result['segment_id'] = f"{segment['speaker']}_{segment['start']:.1f}"
            default_result['segment_duration'] = segment['end'] - segment['start']
            
            return default_result
            
        except Exception as e:
            logger.error(f"Error evaluating segment quality: {e}")
            # Return default scores on error
            return {
                "text_completeness": 5.0,
                "duration_appropriateness": 5.0,
                "speaker_consistency": 5.0,
                "transcription_quality": 5.0,
                "overall_confidence": 5.0,
                "issues": [f"Error in evaluation: {str(e)}"],
                "feedback": f"Evaluation failed: {str(e)}",
                "segment_id": f"{segment['speaker']}_{segment['start']:.1f}",
                "segment_duration": segment['end'] - segment['start']
            }
    
    def evaluate_speaker_transitions(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate quality of speaker transitions
        """
        try:
            if len(segments) < 2:
                return {
                    "transition_quality": 10.0,
                    "num_transitions": 0,
                    "rapid_switches": 0,
                    "large_gaps": 0,
                    "issues": [],
                    "feedback": "Single or no segments - no transitions to evaluate"
                }
            
            transitions = []
            for i in range(len(segments) - 1):
                current = segments[i]
                next_seg = segments[i + 1]
                
                transition = {
                    "from_speaker": current['speaker'],
                    "to_speaker": next_seg['speaker'],
                    "gap": next_seg['start'] - current['end'],
                    "from_text": current['text'][-50:],  # Last 50 chars
                    "to_text": next_seg['text'][:50],   # First 50 chars
                }
                transitions.append(transition)
            
            # Create summary for LLM (limit to avoid token limits)
            transition_summary = "\n".join([
                f"T{i+1}: {t['from_speaker']} -> {t['to_speaker']}, gap: {t['gap']:.2f}s"
                for i, t in enumerate(transitions[:8])  # Limit to 8 transitions
            ])
            
            if len(transitions) > 8:
                transition_summary += f"\n... and {len(transitions) - 8} more transitions"
            
            prompt = f"""
            Analyze speaker transitions for quality:
            
            Total transitions: {len(transitions)}
            
            Sample transitions:
            {transition_summary}
            
            Evaluate transition quality and respond ONLY in this JSON format:
            {{
                "transition_quality": 7.5,
                "rapid_switches": 2,
                "large_gaps": 1,
                "issues": ["issue 1", "issue 2"],
                "feedback": "Brief feedback"
            }}
            """
            
            content = self._make_llm_request(prompt, max_tokens=400)
            result = self._extract_json_from_response(content)
            
            # Ensure required fields
            default_result = {
                "transition_quality": 5.0,
                "rapid_switches": 0,
                "large_gaps": 0,
                "issues": [],
                "feedback": "Unable to analyze transitions"
            }
            
            for key, value in result.items():
                if key in default_result:
                    default_result[key] = value
            
            default_result['num_transitions'] = len(transitions)
            
            return default_result
            
        except Exception as e:
            logger.error(f"Error evaluating transitions: {e}")
            return {
                "transition_quality": 5.0,
                "num_transitions": len(segments) - 1 if len(segments) > 1 else 0,
                "rapid_switches": 0,
                "large_gaps": 0,
                "issues": [f"Error in evaluation: {str(e)}"],
                "feedback": f"Transition evaluation failed: {str(e)}"
            }
    
    def evaluate_speaker_consistency(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate consistency of speaker labeling
        """
        try:
            # Group segments by speaker
            speaker_groups = {}
            for segment in segments:
                speaker = segment['speaker']
                if speaker not in speaker_groups:
                    speaker_groups[speaker] = []
                speaker_groups[speaker].append(segment)
            
            # Analyze each speaker (limit to avoid token limits)
            speaker_analysis = {}
            for speaker, speaker_segments in list(speaker_groups.items())[:5]:  # Limit to 5 speakers
                total_duration = sum([s['end'] - s['start'] for s in speaker_segments])
                avg_confidence = np.mean([s.get('confidence', 0.5) for s in speaker_segments])
                
                speaker_analysis[speaker] = {
                    "segment_count": len(speaker_segments),
                    "total_duration": total_duration,
                    "avg_confidence": avg_confidence,
                }
            
            # Create compact summary
            speaker_summary = "\n".join([
                f"{speaker}: {info['segment_count']} segs, {info['total_duration']:.1f}s, conf: {info['avg_confidence']:.2f}"
                for speaker, info in speaker_analysis.items()
            ])
            
            prompt = f"""
            Analyze speaker consistency:
            
            Total speakers: {len(speaker_groups)}
            
            {speaker_summary}
            
            Respond ONLY in this JSON format:
            {{
                "consistency_score": 7.5,
                "speaker_count": {len(speaker_groups)},
                "labeling_quality": 8.0,
                "balance_score": 6.5,
                "issues": ["issue 1"],
                "feedback": "Brief feedback"
            }}
            """
            
            content = self._make_llm_request(prompt, max_tokens=300)
            result = self._extract_json_from_response(content)
            
            # Ensure required fields
            default_result = {
                "consistency_score": 5.0,
                "speaker_count": len(speaker_groups),
                "labeling_quality": 5.0,
                "balance_score": 5.0,
                "issues": [],
                "feedback": "Unable to analyze consistency"
            }
            
            for key, value in result.items():
                if key in default_result:
                    default_result[key] = value
            
            default_result['speaker_analysis'] = speaker_analysis
            
            return default_result
            
        except Exception as e:
            logger.error(f"Error evaluating speaker consistency: {e}")
            return {
                "consistency_score": 5.0,
                "speaker_count": len(set([s['speaker'] for s in segments])),
                "labeling_quality": 5.0,
                "balance_score": 5.0,
                "issues": [f"Error in evaluation: {str(e)}"],
                "feedback": f"Consistency evaluation failed: {str(e)}",
                "speaker_analysis": {}
            }
    
    def generate_overall_report(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report with progress logging
        """
        logger.info("Generating comprehensive quality report...")
        
        if not segments:
            logger.warning("No segments provided for quality evaluation")
            return {
                "overall_score": 0.0,
                "confidence_level": "LOW",
                "detailed_scores": {},
                "metadata": {"total_segments": 0, "total_speakers": 0, "total_duration": 0},
                "issues": ["No segments provided"],
                "recommendations": ["Check audio processing pipeline"]
            }
        
        # Evaluate individual segments (sample if too many)
        max_segments_to_evaluate = 20  # Limit to avoid token limits and long processing
        segments_to_evaluate = segments[:max_segments_to_evaluate] if len(segments) > max_segments_to_evaluate else segments
        
        logger.info(f"Evaluating {len(segments_to_evaluate)} segments (of {len(segments)} total)")
        
        segment_evaluations = []
        for i, segment in enumerate(segments_to_evaluate):
            if i % 5 == 0:  # Progress logging
                logger.info(f"Evaluating segment {i+1}/{len(segments_to_evaluate)}")
            
            eval_result = self.evaluate_segment_quality(segment)
            segment_evaluations.append(eval_result)
        
        # Evaluate transitions
        logger.info("Evaluating speaker transitions...")
        transition_eval = self.evaluate_speaker_transitions(segments)
        
        # Evaluate speaker consistency
        logger.info("Evaluating speaker consistency...")
        consistency_eval = self.evaluate_speaker_consistency(segments)
        
        # Calculate overall metrics
        if segment_evaluations:
            avg_text_completeness = np.mean([e['text_completeness'] for e in segment_evaluations])
            avg_duration_appropriateness = np.mean([e['duration_appropriateness'] for e in segment_evaluations])
            avg_speaker_consistency = np.mean([e['speaker_consistency'] for e in segment_evaluations])
            avg_transcription_quality = np.mean([e['transcription_quality'] for e in segment_evaluations])
            avg_overall_confidence = np.mean([e['overall_confidence'] for e in segment_evaluations])
        else:
            avg_text_completeness = 0
            avg_duration_appropriateness = 0
            avg_speaker_consistency = 0
            avg_transcription_quality = 0
            avg_overall_confidence = 0
        
        # Calculate final score (weighted average)
        final_score = (
            avg_text_completeness * 0.25 +
            avg_speaker_consistency * 0.25 +
            avg_transcription_quality * 0.20 +
            transition_eval['transition_quality'] * 0.15 +
            consistency_eval['consistency_score'] * 0.15
        )
        
        # Collect all issues (limit to avoid overwhelming output)
        all_issues = []
        for eval_result in segment_evaluations:
            all_issues.extend(eval_result.get('issues', []))
        all_issues.extend(transition_eval.get('issues', []))
        all_issues.extend(consistency_eval.get('issues', []))
        
        # Remove duplicates and limit count
        unique_issues = list(set(all_issues))[:10]  # Limit to 10 most important issues
        
        report = {
            "overall_score": final_score,
            "confidence_level": "HIGH" if final_score >= 7 else "MEDIUM" if final_score >= 5 else "LOW",
            "detailed_scores": {
                "text_completeness": avg_text_completeness,
                "duration_appropriateness": avg_duration_appropriateness,
                "speaker_consistency": avg_speaker_consistency,
                "transcription_quality": avg_transcription_quality,
                "transition_quality": transition_eval['transition_quality'],
                "labeling_consistency": consistency_eval['consistency_score']
            },
            "metadata": {
                "total_segments": len(segments),
                "evaluated_segments": len(segments_to_evaluate),
                "total_speakers": consistency_eval['speaker_count'],
                "total_duration": max([s['end'] for s in segments]) if segments else 0,
                "avg_segment_confidence": avg_overall_confidence
            },
            "issues": unique_issues,
            "segment_evaluations": segment_evaluations,
            "transition_evaluation": transition_eval,
            "consistency_evaluation": consistency_eval,
            "recommendations": self._generate_recommendations(final_score, unique_issues)
        }
        
        logger.info(f"Quality report complete. Overall score: {final_score:.2f}/10 ({report['confidence_level']})")
        return report
    
    def _generate_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """
        Generate recommendations based on score and issues
        """
        recommendations = []
        issues_text = " ".join(issues).lower()
        
        if score < 4:
            recommendations.append("Quality is poor - consider re-processing with different parameters")
            recommendations.append("Check input audio quality and preprocessing steps")
        elif score < 6:
            recommendations.append("Quality is fair - manual review recommended")
            recommendations.append("Consider adjusting diarization sensitivity")
        elif score < 8:
            recommendations.append("Quality is good - minor improvements possible")
        else:
            recommendations.append("Quality is excellent - proceed with confidence")
        
        # Specific issue-based recommendations
        if "transcription" in issues_text or "quality" in issues_text:
            recommendations.append("Consider using a larger Whisper model for better transcription")
        
        if "rapid" in issues_text or "switch" in issues_text:
            recommendations.append("Merge very short consecutive segments from same speaker")
        
        if "consistency" in issues_text or "labeling" in issues_text:
            recommendations.append("Review speaker labeling for potential confusion between speakers")
        
        if "gap" in issues_text:
            recommendations.append("Check for audio gaps or silence detection parameters")
        
        if "duration" in issues_text:
            recommendations.append("Adjust minimum segment duration parameters")
        
        return recommendations[:5]  # Limit to 5 most relevant recommendations'''

import ollama
import json
import logging
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

class QualityCheckAgent:
    def __init__(self, model_name="llama3.1:latest"):
        """
        Initialize quality check agent with Ollama
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Test connection and model availability
        try:
            models = self.client.list()
            
            # Handle different possible structures of the models response
            available_models = []
            if hasattr(models, 'models'):
                # If models is an object with a 'models' attribute
                model_list = models.models
            elif isinstance(models, dict) and 'models' in models:
                # If models is a dict with 'models' key
                model_list = models['models']
            else:
                # If models is directly a list
                model_list = models
            
            # Extract model names from the list
            for model in model_list:
                if hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict):
                    # Try different possible keys for the model name
                    name = model.get('name') or model.get('model') or model.get('id')
                    if name:
                        available_models.append(name)
                else:
                    # If model is just a string
                    available_models.append(str(model))
            
            logger.info(f"Available Ollama models: {available_models}")
            
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not found. Available: {available_models}")
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"Using {self.model_name} instead")
                else:
                    logger.error("No models available")
                    raise Exception("No Ollama models available")
            
            # Test the model with a simple query
            test_response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Hello, are you working?'}],
                options={'num_predict': 10}
            )
            
            logger.info(f"Model {self.model_name} is ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            logger.warning("Quality assessment will be disabled")
            self.model_name = None
    
    def analyze_segment_quality(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze quality of a single segment
        """
        if not self.model_name:
            return self._fallback_quality_analysis(segment)
        
        try:
            prompt = f"""
            Analyze the quality of this speech segment and provide a detailed assessment:
            
            Speaker: {segment.get('speaker', 'Unknown')}
            Duration: {segment.get('end', 0) - segment.get('start', 0):.1f}s
            Text: "{segment.get('text', '')}"
            Confidence: {segment.get('confidence', 0):.2f}
            
            Please assess the following aspects and provide scores (1-10):
            1. Transcription accuracy (based on text coherence)
            2. Audio clarity (inferred from confidence)
            3. Speaker identification confidence
            4. Segment completeness
            
            Respond in JSON format:
            {{
                "accuracy_score": <1-10>,
                "clarity_score": <1-10>, 
                "speaker_confidence": <1-10>,
                "completeness_score": <1-10>,
                "overall_score": <1-10>,
                "issues": ["list", "of", "issues"],
                "suggestions": ["list", "of", "suggestions"]
            }}
            """
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,
                    'num_predict': 200
                }
            )
            
            # Extract response content
            if hasattr(response, 'message'):
                content = response.message.content
            elif isinstance(response, dict):
                content = response.get('message', {}).get('content', '')
            else:
                content = str(response)
            
            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Could not parse AI response as JSON, using fallback")
                return self._fallback_quality_analysis(segment)
                
        except Exception as e:
            logger.error(f"Error in AI quality analysis: {e}")
            return self._fallback_quality_analysis(segment)
    
    def _fallback_quality_analysis(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback quality analysis when AI is not available
        """
        text = segment.get('text', '').strip()
        confidence = segment.get('confidence', 0)
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        # Simple heuristic scoring
        accuracy_score = min(10, confidence * 10)
        clarity_score = min(10, confidence * 12)  # Slightly higher weight
        speaker_confidence = 8 if segment.get('speaker') else 3
        completeness_score = min(10, len(text) / 10) if text else 1
        
        overall_score = (accuracy_score + clarity_score + speaker_confidence + completeness_score) / 4
        
        issues = []
        if confidence < 0.7:
            issues.append("Low transcription confidence")
        if duration < 1.0:
            issues.append("Very short segment")
        if len(text) < 5:
            issues.append("Very short text")
        if not segment.get('speaker'):
            issues.append("No speaker identified")
        
        suggestions = []
        if confidence < 0.8:
            suggestions.append("Consider audio enhancement")
        if duration < 2.0:
            suggestions.append("Consider merging with adjacent segments")
        
        return {
            "accuracy_score": accuracy_score,
            "clarity_score": clarity_score,
            "speaker_confidence": speaker_confidence,
            "completeness_score": completeness_score,
            "overall_score": overall_score,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def generate_overall_report(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for all segments
        """
        if not segments:
            return {
                "overall_score": 0,
                "confidence_level": "No data",
                "total_segments": 0,
                "segment_analyses": [],
                "summary": "No segments to analyze"
            }
        
        logger.info(f"Analyzing quality of {len(segments)} segments...")
        
        segment_analyses = []
        total_scores = {
            "accuracy_score": 0,
            "clarity_score": 0,
            "speaker_confidence": 0,
            "completeness_score": 0,
            "overall_score": 0
        }
        
        all_issues = []
        all_suggestions = []
        
        # Analyze each segment
        for i, segment in enumerate(segments):
            if i % 10 == 0:  # Log progress every 10 segments
                logger.info(f"Analyzing segment {i+1}/{len(segments)}")
            
            analysis = self.analyze_segment_quality(segment)
            analysis['segment_index'] = i
            analysis['segment_text'] = segment.get('text', '')[:100]  # First 100 chars
            
            segment_analyses.append(analysis)
            
            # Accumulate scores
            for key in total_scores:
                total_scores[key] += analysis.get(key, 0)
            
            # Collect issues and suggestions
            all_issues.extend(analysis.get('issues', []))
            all_suggestions.extend(analysis.get('suggestions', []))
        
        # Calculate averages
        avg_scores = {key: value / len(segments) for key, value in total_scores.items()}
        
        # Determine confidence level
        overall_score = avg_scores['overall_score']
        if overall_score >= 8:
            confidence_level = "Excellent"
        elif overall_score >= 6:
            confidence_level = "Good"
        elif overall_score >= 4:
            confidence_level = "Fair"
        else:
            confidence_level = "Poor"
        
        # Get unique issues and suggestions
        unique_issues = list(set(all_issues))
        unique_suggestions = list(set(all_suggestions))
        
        # Generate summary
        total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in segments)
        unique_speakers = len(set(s.get('speaker', 'Unknown') for s in segments))
        
        summary = f"""
        Analyzed {len(segments)} segments ({total_duration:.1f}s total, {unique_speakers} speakers)
        Overall Quality: {overall_score:.2f}/10 ({confidence_level})
        
        Key Metrics:
        - Transcription Accuracy: {avg_scores['accuracy_score']:.1f}/10
        - Audio Clarity: {avg_scores['clarity_score']:.1f}/10
        - Speaker Identification: {avg_scores['speaker_confidence']:.1f}/10
        - Segment Completeness: {avg_scores['completeness_score']:.1f}/10
        
        Common Issues: {', '.join(unique_issues[:3]) if unique_issues else 'None detected'}
        Top Suggestions: {', '.join(unique_suggestions[:3]) if unique_suggestions else 'None'}
        """.strip()
        
        report = {
            "overall_score": overall_score,
            "confidence_level": confidence_level,
            "total_segments": len(segments),
            "total_duration": total_duration,
            "unique_speakers": unique_speakers,
            "average_scores": avg_scores,
            "segment_analyses": segment_analyses,
            "common_issues": unique_issues,
            "suggestions": unique_suggestions,
            "summary": summary,
            "analysis_timestamp": time.time(),
            "ai_model_used": self.model_name or "Fallback heuristics"
        }
        
        logger.info(f"Quality analysis complete. Overall score: {overall_score:.2f}/10 ({confidence_level})")
        
        return report