#!/usr/bin/env python3
"""
Phase CXLVIII: Causal Intervention + Large-Sample Validation + Cross-Layer Tracking (P653-P655)
=================================================================================================

Based on Phase CXLVII findings:
1. k=30 is truncation artifact (like k=300 for ratio=0.22)
2. Hook decomposition is exact (error<0.000002)
3. Three models use completely different emergence strategies
4. Ridge on h(L-1) overfits (training r>0.9 but LOO r<0.5)

This phase addresses:
P653: Large-Sample Validation (200 texts) + Adaptive k Selection
P654: Causal Intervention - scale Attn/MLP/LN outputs, test causality
P655: Cross-Layer Hook Tracking - trace gap information across ALL layers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import argparse
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

# ===== Extended text corpus for large-sample validation =====
TEST_TEXTS_BASIC = [
    "The quantum computer solved the complex optimization problem in seconds.",
    "She walked through the ancient forest, listening to birds singing.",
    "The stock market crashed after the central bank raised interest rates.",
    "Artificial intelligence is transforming healthcare diagnostics.",
    "The musician played a haunting melody on the violin.",
    "Climate change threatens coastal cities with rising sea levels.",
    "The philosopher questioned the nature of consciousness and free will.",
    "A new vaccine was developed to combat the emerging virus.",
    "The architect designed a sustainable building with solar panels.",
    "The detective gathered clues to solve the mysterious case.",
    "The chef prepared a delicious meal using local ingredients.",
    "The spacecraft orbited Mars, collecting data for scientists.",
    "The poet wrote verses about love and loss under moonlight.",
    "The economist predicted a recession based on market indicators.",
    "The teacher encouraged students to think critically about history.",
    "The programmer debugged the code to fix the memory leak.",
    "The artist painted a vibrant landscape with bold brushstrokes.",
    "The biologist discovered a new species in the rainforest.",
    "The judge ruled in favor of the plaintiff after hearing arguments.",
    "The engineer built a bridge that could withstand earthquakes.",
    "The novelist crafted a story about time travel and paradox.",
    "The doctor diagnosed the patient with a rare genetic disorder.",
    "The astronaut floated weightlessly in the International Space Station.",
    "The historian analyzed primary sources from the Renaissance.",
    "The mathematician proved a theorem about prime numbers.",
    "The chemist synthesized a new compound in the laboratory.",
    "The diplomat negotiated a peace agreement between nations.",
    "The journalist reported on the election results from the capital.",
    "The firefighter rescued a family from the burning building.",
    "The musician composed a symphony that moved the audience to tears.",
    "The researcher published a groundbreaking paper on dark matter.",
    "The sailor navigated through the storm using celestial observations.",
    "The developer created an app that simplifies project management.",
    "The photographer captured the sunset over the mountain range.",
    "The botanist studied the rare orchid in its natural habitat.",
    "The librarian organized the archives for public access.",
    "The cyclist trained rigorously for the upcoming championship.",
    "The geologist identified a new mineral formation in the cave.",
    "The translator rendered the ancient text into modern language.",
    "The volunteer distributed food to families affected by the flood.",
]

# Additional texts for large-sample validation
TEST_TEXTS_EXTENDED = [
    "The orchestra performed Beethoven's symphony to a standing ovation.",
    "The student solved the differential equation using integration by parts.",
    "The mechanic repaired the engine after diagnosing the faulty spark plug.",
    "The farmer harvested the wheat before the autumn rain arrived.",
    "The lawyer argued the case before the Supreme Court justices.",
    "The psychologist studied the effects of meditation on anxiety.",
    "The astronaut observed the comet through the telescope's lens.",
    "The chef seasoned the soup with herbs from the garden.",
    "The programmer implemented the algorithm in Python and Rust.",
    "The architect renovated the historic building while preserving its character.",
    "The physicist calculated the trajectory of the particle accelerator beam.",
    "The nurse administered the medication according to the prescription.",
    "The painter applied thin layers of oil paint to the canvas.",
    "The author revised the manuscript based on the editor's feedback.",
    "The pilot navigated through turbulence during the transatlantic flight.",
    "The linguist analyzed the syntax of the ancient Sumerian text.",
    "The environmentalist campaigned against deforestation in the Amazon.",
    "The baker prepared croissants using traditional French techniques.",
    "The philosopher debated the ethics of artificial intelligence.",
    "The musician tuned the piano before the recital began.",
    "The surgeon performed the delicate operation with precision and care.",
    "The engineer designed the circuit board for the satellite communication system.",
    "The mathematician discovered a new pattern in the Fibonacci sequence.",
    "The oceanographer mapped the underwater volcanic activity near Hawaii.",
    "The sociologist conducted interviews to study urban migration patterns.",
    "The astronomer detected a signal from a distant galaxy cluster.",
    "The pharmacist compound the medication according to the doctor's orders.",
    "The sculptor carved the marble statue over several months.",
    "The meteorologist predicted thunderstorms for the weekend forecast.",
    "The therapist helped the patient process traumatic memories.",
    "The electrician installed new wiring throughout the old house.",
    "The choreographer designed the dance sequence for the Broadway show.",
    "The geneticist identified the mutation responsible for the rare disease.",
    "The plumber fixed the leaky pipe under the kitchen sink.",
    "The archaeologist excavated the Roman ruins near the riverbank.",
    "The therapist recommended cognitive behavioral techniques for stress management.",
    "The violinist practiced the concerto for hours each day.",
    "The botanist classified the newly discovered fern species.",
    "The carpenter built the bookshelf from reclaimed oak wood.",
    "The neuroscientist studied the brain activity during meditation.",
    "The financial analyst predicted the market trend using statistical models.",
    "The seamstress altered the wedding dress to fit perfectly.",
    "The paleontologist discovered dinosaur fossils in the sedimentary rock.",
    "The journalist investigated the corruption scandal involving the mayor.",
    "The pharmacist verified the drug interactions before dispensing.",
    "The meteorologist tracked the hurricane as it approached the coast.",
    "The gymnast performed a flawless routine on the balance beam.",
    "The volcanologist monitored the seismic activity near the eruption site.",
    "The professor delivered a lecture on quantum entanglement.",
    "The veterinarian treated the injured bird with antibiotics.",
    "The conductor led the orchestra through Mahler's fifth symphony.",
    "The statistician analyzed the survey data using Bayesian methods.",
    "The potter shaped the clay on the wheel into a vase.",
    "The ecologist studied the symbiotic relationship between the species.",
    "The mechanic replaced the transmission fluid in the vintage car.",
    "The judge sentenced the defendant to five years in prison.",
    "The comedian delivered a hilarious routine about modern technology.",
    "The curator organized the exhibition featuring contemporary artists.",
    "The neurologist examined the patient for signs of multiple sclerosis.",
    "The baker decorated the cake with intricate frosting designs.",
    "The ornithologist observed the migration patterns of Arctic terns.",
    "The locksmith crafted a new key for the antique cabinet.",
    "The radiologist interpreted the MRI scan for abnormalities.",
    "The florist arranged the roses and lilies for the wedding ceremony.",
    "The anthropologist documented the rituals of the indigenous tribe.",
    "The electrician upgraded the circuit breaker for the new appliance.",
    "The pathologist examined the tissue sample under the microscope.",
    "The coach motivated the team before the championship game.",
    "The seismologist recorded the earthquake activity along the fault line.",
    "The optometrist prescribed corrective lenses for the patient.",
    "The tailor measured the customer for a custom suit.",
    "The ichthyologist studied the breeding habits of tropical fish.",
    "The cardiologist recommended lifestyle changes to prevent heart disease.",
    "The weaver created a tapestry depicting scenes from medieval life.",
    "The criminologist analyzed the evidence to understand the criminal's motive.",
    "The dietitian designed a nutrition plan for the athlete's training.",
    "The audiologist tested the patient's hearing using pure tones.",
    "The geographer mapped the population density across the region.",
    "The zoologist documented the behavior of wolves in their natural habitat.",
    "The poet composed a sonnet about the changing seasons.",
    "The engineer calibrated the sensors for the autonomous vehicle.",
    "The dermatologist diagnosed the skin condition as eczema.",
    "The glassblower shaped the molten glass into an ornamental vase.",
    "The hydrologist monitored the water quality in the reservoir.",
    "The ethicist examined the moral implications of gene editing.",
    "The metallurgist tested the alloy for structural integrity.",
    "The pulmonologist treated the patient for chronic obstructive pulmonary disease.",
    "The calligrapher inscribed the invitation with elegant script.",
    "The oceanographer measured the salinity of the deep sea water.",
    "The immunologist researched the body's response to the new pathogen.",
    "The mason constructed the stone wall using traditional techniques.",
    "The orthodontist adjusted the braces to align the patient's teeth.",
    "The glaciologist drilled ice cores to study ancient climate patterns.",
    "The taxonomist classified the insect according to its morphological features.",
    "The urologist performed the procedure to remove the kidney stone.",
    "The silversmith crafted the necklace with intricate filigree work.",
    "The volcanologist predicted the eruption based on gas emissions.",
    "The endocrinologist treated the patient for thyroid dysfunction.",
    "The lexicographer updated the dictionary with newly coined terms.",
    "The sound engineer mixed the tracks for the album release.",
    "The numismatist authenticated the rare coin from the Roman Empire.",
    "The hepatologist evaluated the liver function test results.",
    "The cartographer updated the map with the recently discovered islands.",
    "The cinematographer captured the scene using natural lighting.",
    "The rhematologist prescribed medication for the autoimmune disorder.",
    "The perfumer blended the essential oils to create a new fragrance.",
    "The cryptographer decoded the message using frequency analysis.",
    "The hematologist analyzed the blood sample for abnormalities.",
    "The watchmaker repaired the antique clock with precision tools.",
    "The set designer built the stage for the theatrical production.",
    "The pharmacologist studied the drug's mechanism of action in cells.",
    "The engraver etched the design onto the copper plate.",
    "The oncologist recommended a combination of chemotherapy and radiation.",
    "The sommelier paired the wine with the gourmet cheese selection.",
    "The epidemiologist tracked the spread of the infectious disease.",
    "The upholsterer restored the antique chair with new fabric.",
    "The toxicologist identified the poison in the laboratory sample.",
    "The choreographer rehearsed the dancers for the upcoming performance.",
    "The podiatrist treated the patient's foot condition with orthotics.",
    "The instrument maker crafted the violin from spruce and maple.",
    "The parasitologist studied the life cycle of the malaria parasite.",
    "The audiophile calibrated the speakers for optimal sound quality.",
    "The neurosurgeon performed the operation to remove the brain tumor.",
    "The beekeeper harvested the honey from the apiary.",
    "The virologist developed the vaccine candidate in the biosafety lab.",
    "The leatherworker tooled the belt with an intricate pattern.",
    "the gastroenterologist performed the endoscopy to examine the stomach lining.",
    "The ichthyologist tagged the sharks to track their migration.",
    "The luthier adjusted the soundpost to improve the cello's tone.",
    "The nephrologist managed the dialysis treatment for the patient.",
    "The woodworker joined the planks using dovetail joints.",
    "The ophthalmologist performed the laser surgery to correct the vision.",
    "The entomologist classified the beetle using dichotomous keys.",
    "The typesetter arranged the movable type for the letterpress.",
    "The pulmonologist administered the pulmonary function test.",
    "The ceramist glazed the pottery before the final firing.",
    "The cardiologist inserted the stent to open the blocked artery.",
    "The mycologist identified the mushroom species in the forest.",
    "The bookbinder restored the antique manuscript with care.",
    "The rheumatologist diagnosed the patient with lupus.",
    "The blacksmith forged the iron gate with decorative scrollwork.",
    "The botanist pressed the specimen for the herbarium collection.",
    "The dentist filled the cavity with composite resin.",
    "The upholsterer attached the webbing to the chair frame.",
    "The endocrinologist adjusted the insulin dosage for the diabetic patient.",
    "The lapidary cut the gemstone to maximize its brilliance.",
    "The hematologist diagnosed the blood disorder from the test results.",
    "The farrier shod the horse with custom-fitted iron shoes.",
    "The neurologist ordered the EEG to evaluate the seizure activity.",
    "The weaver dyed the yarn with natural plant extracts.",
    "The oncologist discussed the treatment options with the patient's family.",
    "The clockmaker adjusted the pendulum for accurate timekeeping.",
    "The virologist sequenced the genome of the novel virus.",
    "The cobbler resoled the boots with durable leather.",
    "The immunologist administered the allergy test to identify triggers.",
    "The painter mixed the pigment with linseed oil for the portrait.",
    "The epidemiologist modeled the transmission rate of the outbreak.",
    "The printer aligned the plates for the color reproduction.",
    "The psychiatrist prescribed an SSRI for the depressive disorder.",
    "The beekeeper inspected the hive for signs of disease.",
    "The radiologist identified the fracture on the X-ray image.",
    "The sculptor applied the patina to the bronze statue.",
    "The neurosurgeon mapped the brain areas before the craniotomy.",
    "The baker kneaded the dough until it reached the right consistency.",
    "The pharmacist counseled the patient on the medication side effects.",
    "The mason pointed the brickwork with fresh mortar.",
    "The psychologist assessed the child for learning disabilities.",
    "The glassblower annealed the glass to remove internal stresses.",
    "The surgeon sutured the incision with absorbable stitches.",
    "The botanist measured the growth rate of the hybrid plants.",
    "The optician fitted the frames to the patient's face shape.",
    "The welder joined the steel beams for the bridge construction.",
    "The dermatologist removed the mole using cryotherapy.",
    "The chocolatier tempered the chocolate for the truffle filling.",
    "The astronomer calculated the orbital period of the exoplanet.",
    "The carpenter installed the hardwood floor in the living room.",
    "The pathologist confirmed the diagnosis from the biopsy results.",
    "The barber trimmed the hair with precise scissor technique.",
    "The cardiologist monitored the ECG during the stress test.",
    "The plumber installed the water heater according to building codes.",
    "The naturopath recommended herbal remedies for the digestive issues.",
    "The electrician wired the outlet with the correct polarity.",
    "The orthopedic surgeon replaced the hip joint with a prosthetic.",
    "The brewer monitored the fermentation temperature of the ale.",
    "The radiologist administered the contrast agent for the CT scan.",
    "The tailor pressed the suit with a steam iron.",
    "The dietitian calculated the macronutrient ratios for the meal plan.",
    "The roofer replaced the damaged shingles after the storm.",
    "The dermatologist performed the skin biopsy on the suspicious mole.",
    "The mechanic aligned the wheels to improve the vehicle's handling.",
    "The therapist used exposure therapy to treat the phobia.",
    "The painter applied a coat of primer before the final color.",
    "The surgeon placed the titanium plate to stabilize the fracture.",
    "The baker proofed the bread dough in the warm kitchen.",
    "The veterinarian vaccinated the puppy according to the schedule.",
    "The electrician installed the ceiling fan with the remote control.",
    "The pharmacist checked for drug interactions in the patient's profile.",
    "The mason reinforced the foundation with steel rebar.",
    "The psychologist administered the cognitive assessment battery.",
    "The mechanic replaced the alternator in the sedan.",
    "The dentist cleaned the teeth and applied fluoride treatment.",
    "The cardiologist interpreted the echocardiogram results.",
]

ALL_TEST_TEXTS = TEST_TEXTS_BASIC + TEST_TEXTS_EXTENDED


def compute_features_with_hooks(model, tokenizer, device, texts, n_samples=None):
    """
    Extract features using PyTorch hooks to get EXACT intermediate states.
    Enhanced with cross-layer tracking for ALL layers (not just last).
    """
    if n_samples is not None:
        texts = texts[:n_samples]
    
    # Get W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    
    # Get final LN weights
    final_ln_weight = None
    final_ln_bias = None
    if hasattr(model.model, 'norm'):
        final_ln_weight = model.model.norm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.norm.bias.detach().cpu().float().numpy() if hasattr(model.model.norm, 'bias') and model.model.norm.bias is not None else np.zeros(d_model)
    elif hasattr(model.model, 'final_layernorm'):
        final_ln_weight = model.model.final_layernorm.weight.detach().cpu().float().numpy()
        final_ln_bias = model.model.final_layernorm.bias.detach().cpu().float().numpy() if hasattr(model.model.final_layernorm, 'bias') and model.model.final_layernorm.bias is not None else np.zeros(d_model)
    
    layers = model.model.layers if hasattr(model.model, 'layers') else []
    last_layer = layers[-1]
    
    all_features = []
    
    for i, text in enumerate(texts):
        if (i+1) % 20 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Hook storage for last layer
        hook_data = {}
        
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_data[name] = output[0].detach()
                else:
                    hook_data[name] = output.detach()
            return hook_fn
        
        # Register hooks on the last layer
        handles = []
        handles.append(last_layer.self_attn.register_forward_hook(make_hook('attn_output')))
        handles.append(last_layer.mlp.register_forward_hook(make_hook('mlp_output')))
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('post_attn_ln_input')))
                break
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(last_layer, ln_name):
                handles.append(getattr(last_layer, ln_name).register_forward_hook(make_hook('input_ln_output')))
                break
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Extract ALL hidden states (for cross-layer tracking)
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0, -1, :].cpu().float().numpy())
        
        h_L = all_hidden[-1]
        h_L_minus_1 = all_hidden[-2] if len(all_hidden) > 1 else all_hidden[0]
        
        # Extract hook data
        h_attn_raw = hook_data.get('attn_output', None)
        h_mlp_raw = hook_data.get('mlp_output', None)
        h_input_ln = hook_data.get('input_ln_output', None)
        h_post_attn_ln = hook_data.get('post_attn_ln_input', None)
        
        if h_attn_raw is not None:
            h_attn_raw = h_attn_raw[0, -1, :].cpu().float().numpy()
        if h_mlp_raw is not None:
            h_mlp_raw = h_mlp_raw[0, -1, :].cpu().float().numpy()
        if h_input_ln is not None:
            h_input_ln = h_input_ln[0, -1, :].cpu().float().numpy()
        if h_post_attn_ln is not None:
            h_post_attn_ln = h_post_attn_ln[0, -1, :].cpu().float().numpy()
        
        attn_output = h_attn_raw
        mlp_output = h_mlp_raw
        
        h_attn_out = h_L_minus_1 + (attn_output if attn_output is not None else 0)
        h_mlp_out = h_attn_out + (mlp_output if mlp_output is not None else 0)
        
        # Logits
        logits = outputs.logits[0, -1, :].cpu().float().numpy()
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = logits[top1_idx] - logits[top2_idx]
        
        W_U_top1 = W_U[top1_idx]
        W_U_top2 = W_U[top2_idx]
        Delta_W = W_U_top1 - W_U_top2
        
        # Gap at each stage
        gap_L_minus_1 = np.dot(h_L_minus_1, Delta_W)
        gap_attn_out = np.dot(h_attn_out, Delta_W)
        gap_mlp_out = np.dot(h_mlp_out, Delta_W)
        gap_L = np.dot(h_L, Delta_W)
        
        # Cross-layer gaps
        layer_gaps = []
        for h_layer in all_hidden:
            layer_gaps.append(np.dot(h_layer, Delta_W))
        
        delta_gap_attn = gap_attn_out - gap_L_minus_1
        delta_gap_mlp = gap_mlp_out - gap_attn_out
        delta_gap_final_ln = gap_L - gap_mlp_out
        delta_gap_total = gap_L - gap_L_minus_1
        
        all_features.append({
            'h_L': h_L,
            'h_L_minus_1': h_L_minus_1,
            'h_attn_out': h_attn_out,
            'h_mlp_out': h_mlp_out,
            'attn_output': attn_output,
            'mlp_output': mlp_output,
            'h_input_ln': h_input_ln,
            'h_post_attn_ln': h_post_attn_ln,
            'logit_gap': logit_gap,
            'top1_idx': int(top1_idx),
            'top2_idx': int(top2_idx),
            'Delta_W': Delta_W,
            'gap_L_minus_1': gap_L_minus_1,
            'gap_attn_out': gap_attn_out,
            'gap_mlp_out': gap_mlp_out,
            'gap_L': gap_L,
            'delta_gap_attn': delta_gap_attn,
            'delta_gap_mlp': delta_gap_mlp,
            'delta_gap_final_ln': delta_gap_final_ln,
            'delta_gap_total': delta_gap_total,
            'layer_gaps': layer_gaps,
            'all_hidden': all_hidden,
            'text_idx': i,
        })
    
    extra = {
        'W_U': W_U,
        'd_model': d_model,
        'n_layers': n_layers,
        'final_ln_weight': final_ln_weight,
        'final_ln_bias': final_ln_bias,
    }
    
    return all_features, extra


def experiment_p653(all_features, extra, model_name):
    """P653: Large-Sample Validation + Adaptive k Selection.
    
    Key questions:
    1. Does Ridge h(L-1)->gap survive LOO-CV with 200 texts?
    2. What is the optimal k as a function of sample size?
    3. Does gap(h(L-1)) correlation with logit_gap improve with more data?
    """
    print(f"\n{'='*60}")
    print(f"P653: Large-Sample Validation ({model_name}, n={len(all_features)})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    
    h_L_minus_1 = np.array([f['h_L_minus_1'] for f in all_features])
    h_L = np.array([f['h_L'] for f in all_features])
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    
    # ===== 1. Sample size scaling =====
    print(f"\n1. Sample Size Scaling (Ridge LOO with adaptive k):")
    
    sample_sizes = [20, 40, 80, 120, 160, n_texts]
    scaling_results = []
    
    for n_sample in sample_sizes:
        if n_sample > n_texts:
            n_sample = n_texts
        
        # Random subset (seed for reproducibility)
        np.random.seed(42)
        indices = np.random.choice(n_texts, n_sample, replace=False)
        h_sub = h_L_minus_1[indices]
        gaps_sub = gaps[indices]
        
        # Adaptive k: use min(n_sample-1, 50) but scan for optimal
        max_k = min(n_sample - 1, 60, d_model)
        best_k = 1
        best_r = -1
        
        # Quick scan (skip some k for speed)
        k_candidates = list(range(1, min(10, max_k+1)))
        k_candidates += list(range(10, max_k+1, 5))
        if max_k not in k_candidates:
            k_candidates.append(max_k)
        
        for k in k_candidates:
            try:
                pca = PCA(n_components=k)
                h_pca = pca.fit_transform(h_sub)
                
                # LOO-CV
                loo_preds = []
                for train_idx, test_idx in LeaveOneOut().split(h_pca):
                    ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps_sub[train_idx])
                    loo_preds.append(ridge.predict(h_pca[test_idx])[0])
                loo_preds = np.array(loo_preds)
                r_loo, _ = stats.pearsonr(loo_preds, gaps_sub)
                
                if r_loo > best_r:
                    best_r = r_loo
                    best_k = k
            except Exception as e:
                pass
        
        # Also compute gap(h(L-1), Delta_W) correlation
        gap_Lm1_sub = np.array([np.dot(h_L_minus_1[idx], Delta_Ws[idx]) for idx in indices])
        r_gap_Lm1, _ = stats.pearsonr(gap_Lm1_sub, gaps_sub)
        
        # And Ridge with LOO using best k
        pca = PCA(n_components=best_k)
        h_pca = pca.fit_transform(h_sub)
        loo_preds = []
        for train_idx, test_idx in LeaveOneOut().split(h_pca):
            ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps_sub[train_idx])
            loo_preds.append(ridge.predict(h_pca[test_idx])[0])
        loo_preds = np.array(loo_preds)
        r_loo_best, _ = stats.pearsonr(loo_preds, gaps_sub)
        
        # Training set r (for comparison)
        ridge_full = Ridge(alpha=1.0).fit(h_pca, gaps_sub)
        r_train, _ = stats.pearsonr(ridge_full.predict(h_pca), gaps_sub)
        
        scaling_results.append({
            'n': n_sample, 'best_k': best_k,
            'r_loo': r_loo_best, 'r_train': r_train,
            'r_gap_Lm1': r_gap_Lm1,
        })
        print(f"   n={n_sample:3d}: best_k={best_k:2d}, r_LOO={r_loo_best:.4f}, "
              f"r_train={r_train:.4f}, gap(L-1) r={r_gap_Lm1:.4f}")
    
    # ===== 2. Overfitting diagnosis =====
    print(f"\n2. Overfitting Diagnosis:")
    for res in scaling_results:
        overfit_gap = res['r_train'] - res['r_loo']
        print(f"   n={res['n']:3d}: r_train - r_LOO = {overfit_gap:.4f} "
              f"({'SEVERE overfit' if overfit_gap > 0.3 else 'moderate overfit' if overfit_gap > 0.1 else 'OK'})")
    
    # ===== 3. Hook decomposition at large scale =====
    print(f"\n3. Hook Decomposition at n={n_texts}:")
    
    gap_L_minus_1 = np.array([f['gap_L_minus_1'] for f in all_features])
    delta_attn = np.array([f['delta_gap_attn'] for f in all_features])
    delta_mlp = np.array([f['delta_gap_mlp'] for f in all_features])
    delta_final_ln = np.array([f['delta_gap_final_ln'] for f in all_features])
    delta_total = np.array([f['delta_gap_total'] for f in all_features])
    
    r_gap_Lm1, _ = stats.pearsonr(gap_L_minus_1, gaps)
    
    sign_attn = np.mean(np.sign(delta_attn) == np.sign(delta_total))
    sign_mlp = np.mean(np.sign(delta_mlp) == np.sign(delta_total))
    sign_ln = np.mean(np.sign(delta_final_ln) == np.sign(delta_total))
    
    var_attn = np.var(delta_attn) / np.var(delta_total) * 100
    var_mlp = np.var(delta_mlp) / np.var(delta_total) * 100
    var_ln = np.var(delta_final_ln) / np.var(delta_total) * 100
    
    print(f"   gap(h(L-1)) -> logit_gap: r={r_gap_Lm1:.4f}")
    print(f"   sign(Attn)={sign_attn:.3f}, sign(MLP)={sign_mlp:.3f}, sign(LN)={sign_ln:.3f}")
    print(f"   var%: Attn={var_attn:.1f}%, MLP={var_mlp:.1f}%, LN={var_ln:.1f}%")
    
    # Dominant component
    var_dict = {'Attn': var_attn, 'MLP': var_mlp, 'LN': var_ln}
    dominant = max(var_dict, key=var_dict.get)
    print(f"   Dominant: {dominant} ({var_dict[dominant]:.1f}%)")
    
    # ===== 4. Bootstrap CI for sign consistency =====
    print(f"\n4. Bootstrap CI for Sign Consistency (1000 iterations):")
    n_boot = 1000
    np.random.seed(42)
    
    boot_signs = {'Attn': [], 'MLP': [], 'LN': []}
    for _ in range(n_boot):
        idx = np.random.choice(n_texts, n_texts, replace=True)
        d_a = delta_attn[idx]
        d_m = delta_mlp[idx]
        d_l = delta_final_ln[idx]
        d_t = delta_total[idx]
        boot_signs['Attn'].append(np.mean(np.sign(d_a) == np.sign(d_t)))
        boot_signs['MLP'].append(np.mean(np.sign(d_m) == np.sign(d_t)))
        boot_signs['LN'].append(np.mean(np.sign(d_l) == np.sign(d_t)))
    
    for comp in ['Attn', 'MLP', 'LN']:
        vals = np.array(boot_signs[comp])
        ci_low = np.percentile(vals, 2.5)
        ci_high = np.percentile(vals, 97.5)
        print(f"   sign({comp}): {np.mean(vals):.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    return {
        'n_texts': n_texts,
        'scaling_results': scaling_results,
        'dominant': dominant,
        'sign_attn': float(sign_attn),
        'sign_mlp': float(sign_mlp),
        'sign_ln': float(sign_ln),
        'var_attn': float(var_attn),
        'var_mlp': float(var_mlp),
        'var_ln': float(var_ln),
        'r_gap_Lm1': float(r_gap_Lm1),
    }


def experiment_p654(model, tokenizer, device, all_features, extra, model_name):
    """P654: Causal Intervention - scale Attn/MLP/LN outputs.
    
    Key idea: If Hook decomposition shows Attn contributes 51% of gap,
    then scaling Attn output by factor s should change gap by ~s*51%.
    
    This tests whether the correlation is also causation.
    """
    print(f"\n{'='*60}")
    print(f"P654: Causal Intervention ({model_name})")
    print(f"{'='*60}")
    
    d_model = extra['d_model']
    n_layers = extra['n_layers']
    layers = model.model.layers if hasattr(model.model, 'layers') else []
    last_layer = layers[-1]
    
    # Use 20 texts for intervention (computationally expensive)
    test_texts = ALL_TEST_TEXTS[:20]
    scale_factors = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    W_U = extra['W_U']
    
    # ===== Intervention 1: Scale Attn output =====
    print(f"\n1. Scale Attn Output Intervention:")
    print(f"   (Scaling attn_output before residual addition)")
    
    attn_intervention = []
    
    for scale in scale_factors:
        gap_values = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Hook to scale attention output
            hook_data = {}
            
            def make_scale_hook(name, s):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hook_data[name] = output[0].detach()
                        # Scale the output
                        scaled = output[0] * s
                        return (scaled,) + output[1:]
                    else:
                        hook_data[name] = output.detach()
                        return output * s
                return hook_fn
            
            handle = last_layer.self_attn.register_forward_hook(make_scale_hook('attn', scale))
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            handle.remove()
            
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            top1_idx = np.argmax(logits)
            top2_idx = np.argsort(logits)[-2]
            gap = logits[top1_idx] - logits[top2_idx]
            gap_values.append(gap)
        
        gap_arr = np.array(gap_values)
        attn_intervention.append({
            'scale': scale,
            'mean_gap': float(np.mean(gap_arr)),
            'std_gap': float(np.std(gap_arr)),
        })
        print(f"   scale={scale:.2f}: mean_gap={np.mean(gap_arr):.4f}, std={np.std(gap_arr):.4f}")
    
    # ===== Intervention 2: Scale MLP output =====
    print(f"\n2. Scale MLP Output Intervention:")
    
    mlp_intervention = []
    
    for scale in scale_factors:
        gap_values = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            hook_data = {}
            
            def make_scale_hook(name, s):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hook_data[name] = output[0].detach()
                        scaled = output[0] * s
                        return (scaled,) + output[1:]
                    else:
                        hook_data[name] = output.detach()
                        return output * s
                return hook_fn
            
            handle = last_layer.mlp.register_forward_hook(make_scale_hook('mlp', scale))
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            handle.remove()
            
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            top1_idx = np.argmax(logits)
            top2_idx = np.argsort(logits)[-2]
            gap = logits[top1_idx] - logits[top2_idx]
            gap_values.append(gap)
        
        gap_arr = np.array(gap_values)
        mlp_intervention.append({
            'scale': scale,
            'mean_gap': float(np.mean(gap_arr)),
            'std_gap': float(np.std(gap_arr)),
        })
        print(f"   scale={scale:.2f}: mean_gap={np.mean(gap_arr):.4f}, std={np.std(gap_arr):.4f}")
    
    # ===== Intervention 3: Scale Final LN output =====
    print(f"\n3. Scale Final LN Weight Intervention:")
    
    # Get original final LN weight
    if hasattr(model.model, 'norm'):
        orig_ln_weight = model.model.norm.weight.data.clone()
    elif hasattr(model.model, 'final_layernorm'):
        orig_ln_weight = model.model.final_layernorm.weight.data.clone()
    else:
        print("   WARNING: Cannot find final LN, skipping")
        orig_ln_weight = None
    
    ln_intervention = []
    
    if orig_ln_weight is not None:
        for scale in scale_factors:
            # Modify LN weight in-place
            if hasattr(model.model, 'norm'):
                model.model.norm.weight.data = orig_ln_weight * scale
            elif hasattr(model.model, 'final_layernorm'):
                model.model.final_layernorm.weight.data = orig_ln_weight * scale
            
            gap_values = []
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs.logits[0, -1, :].cpu().float().numpy()
                top1_idx = np.argmax(logits)
                top2_idx = np.argsort(logits)[-2]
                gap = logits[top1_idx] - logits[top2_idx]
                gap_values.append(gap)
            
            gap_arr = np.array(gap_values)
            ln_intervention.append({
                'scale': scale,
                'mean_gap': float(np.mean(gap_arr)),
                'std_gap': float(np.std(gap_arr)),
            })
            print(f"   scale={scale:.2f}: mean_gap={np.mean(gap_arr):.4f}, std={np.std(gap_arr):.4f}")
        
        # Restore original weight
        if hasattr(model.model, 'norm'):
            model.model.norm.weight.data = orig_ln_weight.clone()
        elif hasattr(model.model, 'final_layernorm'):
            model.model.final_layernorm.weight.data = orig_ln_weight.clone()
    
    # ===== Analysis: Linearity of intervention response =====
    print(f"\n4. Intervention Linearity Analysis:")
    
    for name, data in [('Attn', attn_intervention), ('MLP', mlp_intervention), ('LN', ln_intervention)]:
        if len(data) < 2:
            continue
        scales = np.array([d['scale'] for d in data])
        mean_gaps = np.array([d['mean_gap'] for d in data])
        
        # Fit linear model on non-zero scales
        mask = scales > 0
        if mask.sum() >= 2:
            slope, intercept, r_val, p_val, se = stats.linregress(scales[mask], mean_gaps[mask])
            print(f"   {name}: slope={slope:.4f}, R2={r_val**2:.4f}, p={p_val:.6f}")
            
            # Check: is the response linear?
            # If truly linear, gap(scale=2.0) should be 2x gap(scale=1.0)
            gap_at_1 = mean_gaps[scales == 1.0][0] if 1.0 in scales else None
            gap_at_2 = mean_gaps[scales == 2.0][0] if 2.0 in scales else None
            if gap_at_1 is not None and gap_at_2 is not None and gap_at_1 != 0:
                ratio_2x = gap_at_2 / gap_at_1
                print(f"   {name}: gap(2x)/gap(1x) = {ratio_2x:.4f} (linear prediction: 2.000)")
    
    return {
        'attn_intervention': attn_intervention,
        'mlp_intervention': mlp_intervention,
        'ln_intervention': ln_intervention,
    }


def experiment_p655(all_features, extra, model_name):
    """P655: Cross-Layer Hook Tracking.
    
    Key questions:
    1. At which layer does gap information start to emerge?
    2. Is gap emergence gradual or sudden?
    3. Which layers show the largest gap transformation?
    """
    print(f"\n{'='*60}")
    print(f"P655: Cross-Layer Gap Tracking ({model_name}, n={len(all_features)})")
    print(f"{'='*60}")
    
    n_texts = len(all_features)
    gaps = np.array([f['logit_gap'] for f in all_features])
    n_layers = extra['n_layers']
    
    # ===== 1. Layer-by-layer gap correlation =====
    print(f"\n1. Layer-by-Layer gap(h_l, Delta_W) -> logit_gap correlation:")
    
    layer_correlations = []
    
    # Get layer_gaps from features
    for layer_idx in range(n_layers + 1):  # +1 for embedding layer
        layer_gaps = np.array([f['layer_gaps'][layer_idx] for f in all_features])
        r, p = stats.pearsonr(layer_gaps, gaps)
        layer_correlations.append({
            'layer': layer_idx,
            'r': r,
            'p': p,
            'mean_gap': float(np.mean(np.abs(layer_gaps))),
            'std_gap': float(np.std(layer_gaps)),
        })
        
        # Print every 5 layers + last few
        if layer_idx % 5 == 0 or layer_idx >= n_layers - 3 or layer_idx == 0:
            label = " (embed)" if layer_idx == 0 else f" (L{layer_idx-1})" if layer_idx > 0 else ""
            if layer_idx == n_layers:
                label = " (final)"
            print(f"   Layer {layer_idx:3d}{label}: r={r:.4f}, p={p:.4e}, |gap|={np.mean(np.abs(layer_gaps)):.4f}")
    
    # ===== 2. Identify critical transition layer =====
    print(f"\n2. Critical Transition Layer:")
    
    # Find the layer where correlation first exceeds 0.5
    first_significant = None
    first_high = None
    for lc in layer_correlations:
        if lc['r'] > 0.1 and first_significant is None:
            first_significant = lc['layer']
        if lc['r'] > 0.5 and first_high is None:
            first_high = lc['layer']
    
    # Find the layer with maximum correlation
    max_r_layer = max(layer_correlations, key=lambda x: x['r'])
    
    # Find the layer with maximum correlation jump
    max_jump = 0
    max_jump_layer = 0
    for i in range(1, len(layer_correlations)):
        jump = layer_correlations[i]['r'] - layer_correlations[i-1]['r']
        if jump > max_jump:
            max_jump = jump
            max_jump_layer = i
    
    print(f"   First r>0.1: Layer {first_significant}")
    print(f"   First r>0.5: Layer {first_high}")
    print(f"   Max r: Layer {max_r_layer['layer']} (r={max_r_layer['r']:.4f})")
    print(f"   Max jump: Layer {max_jump_layer} (Δr={max_jump:.4f})")
    
    # ===== 3. Gradual vs Sudden emergence =====
    print(f"\n3. Gradual vs Sudden Emergence:")
    
    # Look at the last 5 layers
    last_5 = layer_correlations[-5:]
    print(f"   Last 5 layers:")
    for lc in last_5:
        print(f"     Layer {lc['layer']}: r={lc['r']:.4f}")
    
    # Check: is the jump from L-2 to L-1 bigger than L-5 to L-2?
    if len(layer_correlations) >= 5:
        r_Lm2 = layer_correlations[-3]['r']
        r_Lm1 = layer_correlations[-2]['r']
        r_L = layer_correlations[-1]['r']
        
        jump_last = r_L - r_Lm1
        jump_prev = r_Lm1 - r_Lm2
        
        print(f"\n   L-2->L-1 jump: {jump_prev:.4f}")
        print(f"   L-1->L  jump: {jump_last:.4f}")
        print(f"   {'Sudden emergence!' if jump_last > 0.3 else 'Gradual emergence'} at last layer")
    
    # ===== 4. Ridge prediction at each layer =====
    print(f"\n4. Ridge h(l) -> logit_gap LOO-CV at each layer:")
    
    # Sample every 4 layers for speed
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)
    
    ridge_by_layer = []
    
    for layer_idx in sample_layers:
        h_layer = np.array([f['all_hidden'][layer_idx] for f in all_features])
        
        # Adaptive PCA
        max_k = min(n_texts - 1, 40, h_layer.shape[1])
        best_k = 1
        best_r = -1
        
        for k in [1, 3, 5, 10, 15, 20, 30, 40]:
            if k > max_k:
                break
            try:
                pca = PCA(n_components=k)
                h_pca = pca.fit_transform(h_layer)
                
                loo_preds = []
                for train_idx, test_idx in LeaveOneOut().split(h_pca):
                    ridge = Ridge(alpha=1.0).fit(h_pca[train_idx], gaps[train_idx])
                    loo_preds.append(ridge.predict(h_pca[test_idx])[0])
                loo_preds = np.array(loo_preds)
                r_loo, _ = stats.pearsonr(loo_preds, gaps)
                
                if r_loo > best_r:
                    best_r = r_loo
                    best_k = k
            except:
                pass
        
        ridge_by_layer.append({
            'layer': layer_idx,
            'best_k': best_k,
            'r_loo': best_r,
        })
        label = " (embed)" if layer_idx == 0 else f" (L{layer_idx-1})"
        print(f"   Layer {layer_idx:3d}{label}: best_k={best_k:2d}, Ridge LOO r={best_r:.4f}")
    
    # ===== 5. Gap direction alignment across layers =====
    print(f"\n5. Gap Direction Alignment (cos between h_l and Delta_W):")
    
    Delta_Ws = np.array([f['Delta_W'] for f in all_features])
    
    for layer_idx in [0, n_layers//2, n_layers-1, n_layers]:
        h_layer = np.array([f['all_hidden'][layer_idx] for f in all_features])
        # Cosine similarity between h and Delta_W
        cos_vals = []
        for j in range(n_texts):
            h_norm = np.linalg.norm(h_layer[j])
            d_norm = np.linalg.norm(Delta_Ws[j])
            if h_norm > 0 and d_norm > 0:
                cos_vals.append(np.dot(h_layer[j], Delta_Ws[j]) / (h_norm * d_norm))
            else:
                cos_vals.append(0)
        mean_cos = np.mean(cos_vals)
        label = " (embed)" if layer_idx == 0 else f" (L{layer_idx-1})"
        print(f"   Layer {layer_idx:3d}{label}: mean cos(h, ΔW)={mean_cos:.6f}")
    
    return {
        'layer_correlations': layer_correlations,
        'first_significant_layer': first_significant,
        'first_high_layer': first_high,
        'max_r_layer': max_r_layer['layer'],
        'max_jump_layer': max_jump_layer,
        'ridge_by_layer': ridge_by_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--experiment', type=str, required=True, choices=['p653', 'p654', 'p655'])
    parser.add_argument('--n_texts', type=int, default=200, help='Number of texts for P653')
    args = parser.parse_args()
    
    model_name = args.model
    experiment = args.experiment
    
    print(f"\n{'#'*60}")
    print(f"# Phase CXLVIII: {experiment.upper()} on {model_name}")
    print(f"{'#'*60}")
    
    # Load model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.name}, layers={info.n_layers}, d_model={info.d_model}")
    
    # Determine number of texts
    n_texts = min(args.n_texts, len(ALL_TEST_TEXTS))
    if experiment == 'p654':
        n_texts = 40  # P654 only needs 40 texts (hooks already extracted)
    elif experiment == 'p655':
        n_texts = min(args.n_texts, len(ALL_TEST_TEXTS))
    
    print(f"Using {n_texts} texts")
    
    # Extract features with hooks
    print(f"\nExtracting features with hooks...")
    t0 = time.time()
    all_features, extra = compute_features_with_hooks(model, tokenizer, device, ALL_TEST_TEXTS, n_samples=n_texts)
    print(f"Feature extraction: {time.time()-t0:.1f}s, {len(all_features)} samples")
    
    # Run experiment
    if experiment == 'p653':
        result = experiment_p653(all_features, extra, model_name)
    elif experiment == 'p654':
        result = experiment_p654(model, tokenizer, device, all_features, extra, model_name)
    elif experiment == 'p655':
        result = experiment_p655(all_features, extra, model_name)
    
    # Save results
    os.makedirs(f'results/phase_cxlviii', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    result = make_serializable(result)
    
    out_path = f'results/phase_cxlviii/{experiment}_{model_name}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    
    # Release model
    release_model(model)
    print(f"\nModel released. Done!")


if __name__ == '__main__':
    main()
