"""
SAMM Model Tester
Test your fine-tuned model on SAMM queries

Usage:
    python samm_model_tester.py --model ./samm_llama_qlora
"""

import torch
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

# ============================================================================
# TEST QUERIES
# ============================================================================

SAMM_TEST_QUERIES = [
    {
        "query": "What is Security Cooperation?",
        "expected_keywords": ["DoD", "international partners", "strategic objectives", "Title 10"],
        "category": "definition"
    },
    {
        "query": "What is the difference between Security Cooperation and Security Assistance?",
        "expected_keywords": ["subset", "Title 10", "Title 22", "broader", "narrower"],
        "category": "distinction",
        "critical": True  # This must be answered correctly
    },
    {
        "query": "Who supervises Security Assistance programs?",
        "expected_keywords": ["Department of State", "Secretary of State", "supervision"],
        "category": "authority"
    },
    {
        "query": "What does DSCA do?",
        "expected_keywords": ["Defense Security Cooperation Agency", "directs", "administers", "guidance"],
        "category": "organization"
    },
    {
        "query": "What is the role of DFAS in Security Cooperation?",
        "expected_keywords": ["Defense Finance and Accounting Service", "accounting", "billing", "disbursing"],
        "category": "organization"
    },
    {
        "query": "Is Security Assistance broader than Security Cooperation?",
        "expected_keywords": ["no", "incorrect", "subset", "narrower"],
        "category": "distinction",
        "critical": True  # Must not get this wrong!
    }
]

# ============================================================================
# MODEL LOADER
# ============================================================================

class SAMMModelTester:
    """Test fine-tuned SAMM model"""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-2-7b-hf"):
        """Load model for testing"""
        
        print("="*80)
        print("🧪 SAMM Model Tester")
        print("="*80)
        print(f"📦 Base Model: {base_model}")
        print(f"🎯 Fine-tuned Model: {model_path}")
        print("="*80 + "\n")
        
        self.model_path = model_path
        self.base_model = base_model
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print("  └─ Done\n")
        
        # Load model (quantized for efficiency)
        print("🤖 Loading model (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        print("  └─ Model ready\n")
    
    def generate_answer(self, query: str, max_new_tokens: int = 512) -> str:
        """Generate answer for a query"""
        
        # Format prompt
        prompt = f"""<s>[INST] <<SYS>>
You are a SAMM (Security Assistance Management Manual) expert. Provide accurate, concise answers based on SAMM guidance. Always cite section numbers when applicable.
<</SYS>>

{query} [/INST]"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = response.split("[/INST]")[-1].strip()
        
        return answer
    
    def evaluate_answer(self, query: str, answer: str, expected_keywords: List[str]) -> Dict:
        """Evaluate answer quality"""
        
        answer_lower = answer.lower()
        
        # Check for expected keywords
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_score = len(found_keywords) / len(expected_keywords)
        
        # Check length (should be substantial but not excessive)
        word_count = len(answer.split())
        length_ok = 20 <= word_count <= 300
        
        # Check for section citation
        has_citation = "C1." in answer or "Section" in answer or "SAMM" in answer
        
        # Overall quality score
        quality_score = (
            keyword_score * 0.6 +  # 60% weight on keywords
            (1.0 if length_ok else 0.5) * 0.2 +  # 20% on length
            (1.0 if has_citation else 0.5) * 0.2  # 20% on citation
        )
        
        return {
            "keyword_score": keyword_score,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in expected_keywords if kw not in found_keywords],
            "word_count": word_count,
            "has_citation": has_citation,
            "quality_score": quality_score
        }
    
    def run_tests(self, queries: List[Dict] = None) -> Dict:
        """Run all test queries"""
        
        if queries is None:
            queries = SAMM_TEST_QUERIES
        
        print("🧪 Running SAMM Knowledge Tests\n")
        print("="*80)
        
        results = []
        
        for i, test in enumerate(queries, 1):
            query = test['query']
            expected_keywords = test['expected_keywords']
            category = test['category']
            critical = test.get('critical', False)
            
            print(f"\n📝 Test {i}/{len(queries)} [{category}]")
            if critical:
                print("   🔴 CRITICAL TEST")
            print(f"   Query: {query}")
            
            # Generate answer
            answer = self.generate_answer(query)
            
            # Evaluate
            evaluation = self.evaluate_answer(query, answer, expected_keywords)
            
            print(f"\n   Answer: {answer[:200]}...")
            print(f"\n   Evaluation:")
            print(f"     ├─ Quality Score: {evaluation['quality_score']:.2%}")
            print(f"     ├─ Keyword Match: {evaluation['keyword_score']:.2%} ({len(evaluation['found_keywords'])}/{len(expected_keywords)})")
            print(f"     ├─ Word Count: {evaluation['word_count']}")
            print(f"     └─ Has Citation: {'✅' if evaluation['has_citation'] else '❌'}")
            
            if evaluation['missing_keywords']:
                print(f"     ⚠️  Missing keywords: {', '.join(evaluation['missing_keywords'])}")
            
            # Check if critical test passed
            if critical and evaluation['quality_score'] < 0.7:
                print(f"     🔴 CRITICAL TEST FAILED!")
            
            results.append({
                "query": query,
                "answer": answer,
                "category": category,
                "critical": critical,
                **evaluation
            })
            
            print("-"*80)
        
        # Overall summary
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        critical_results = [r for r in results if r.get('critical', False)]
        critical_passed = sum(1 for r in critical_results if r['quality_score'] >= 0.7)
        
        print("\n" + "="*80)
        print("📊 OVERALL RESULTS")
        print("="*80)
        print(f"Average Quality Score: {avg_quality:.2%}")
        print(f"Tests Passed (>70%): {sum(1 for r in results if r['quality_score'] >= 0.7)}/{len(results)}")
        
        if critical_results:
            print(f"Critical Tests Passed: {critical_passed}/{len(critical_results)}")
        
        print("="*80 + "\n")
        
        return {
            "average_quality": avg_quality,
            "tests_passed": sum(1 for r in results if r['quality_score'] >= 0.7),
            "total_tests": len(results),
            "critical_passed": critical_passed if critical_results else None,
            "critical_total": len(critical_results) if critical_results else None,
            "detailed_results": results
        }
    
    def save_results(self, results: Dict, output_path: str = "./test_results.json"):
        """Save test results"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAMM fine-tuned model")
    parser.add_argument("--model", type=str, default="./samm_llama_qlora",
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model name")
    parser.add_argument("--output", type=str, default="./test_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run tests
    tester = SAMMModelTester(args.model, args.base_model)
    results = tester.run_tests()
    tester.save_results(results, args.output)
    
    # Exit with error if critical tests failed
    if results.get('critical_total'):
        if results['critical_passed'] < results['critical_total']:
            print("\n⚠️  WARNING: Some critical tests failed!")
            print("   Consider additional fine-tuning or dataset improvements.")
            exit(1)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
