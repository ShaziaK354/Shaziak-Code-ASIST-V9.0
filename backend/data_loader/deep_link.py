from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
from urllib.parse import urljoin
from collections import deque

print("="*60)
print("DSCA Recursive Deep Link Parser - Starting...")
print("="*60)

# Setup Edge options
edge_options = Options()
# edge_options.add_argument('--headless')
edge_options.add_argument('--start-maximized')

# Path to the Edge driver
driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')

# Check if driver exists
if not os.path.exists(driver_path):
    print(f"\n✗ ERROR: Edge driver not found at: {driver_path}")
    print("\nPlease download the Edge driver and place it in the project folder.")
    exit()

# Initialize the driver
print("\n[1/6] Initializing Edge browser...")
driver = webdriver.Edge(service=Service(driver_path), options=edge_options)

try:
    # Base URL for DSCA
    base_url = "https://samm.dsca.mil"
    
    # Starting point - Appendix 7 overview page
    start_url = f"{base_url}/appendix/appendix-7-case-reconciliation-and-closure-guide-rcg"
    
    print(f"[2/6] Opening starting page: {start_url}")
    driver.get(start_url)
    time.sleep(3)
    
    # Wait for page to load
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    # Data structures for tracking
    all_links = []  # Store all discovered links with metadata
    visited_urls = set()  # Track visited URLs to avoid duplicates
    queue = deque()  # BFS queue for recursive discovery
    
    # Add starting page to queue
    queue.append({
        'url': start_url,
        'level': 0,
        'parent': None,
        'section_id': 'AP7',
        'title': 'Appendix 7 - Case Reconciliation and Closure Guide'
    })
    
    print("[3/6] Discovering all deep links recursively...")
    
    # BFS recursive link discovery
    while queue:
        current = queue.popleft()
        current_url = current['url']
        
        # Skip if already visited
        if current_url in visited_urls:
            continue
            
        visited_urls.add(current_url)
        print(f"\n  Level {current['level']}: Scanning {current['section_id']} - {current['title'][:50]}...")
        
        # Navigate to the page
        driver.get(current_url)
        time.sleep(2)
        
        # Wait for content
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            print(f"    ✗ Timeout loading page")
            continue
        
        # Extract page content
        page_data = {
            'level': current['level'],
            'section_id': current['section_id'],
            'title': current['title'],
            'url': current_url,
            'parent_section': current['parent'],
            'has_subsections': False,
            'content_preview': ''
        }
        
        # Try to extract some content
        try:
            # Look for main content area
            content_elements = driver.find_elements(By.CSS_SELECTOR, "p, div.field--name-body")
            if content_elements:
                text = ' '.join([el.text.strip() for el in content_elements[:3] if el.text.strip()])
                page_data['content_preview'] = text[:200] if text else "No content extracted"
        except:
            page_data['content_preview'] = "Content extraction failed"
        
        # Find subsection links on this page
        # Look for section tables or navigation elements
        subsection_links = []
        
        try:
            # Strategy 1: Find tables with section listings (common pattern)
            tables = driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows[1:]:  # Skip header
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        # First cell usually has section ID, second has title
                        section_cell = cells[0]
                        title_cell = cells[1]
                        
                        # Look for links in either cell
                        link = None
                        if section_cell.find_elements(By.TAG_NAME, "a"):
                            link = section_cell.find_element(By.TAG_NAME, "a")
                        elif title_cell.find_elements(By.TAG_NAME, "a"):
                            link = title_cell.find_element(By.TAG_NAME, "a")
                        
                        if link:
                            href = link.get_attribute("href")
                            section_text = section_cell.text.strip()
                            title_text = title_cell.text.strip()
                            
                            if href and section_text and title_text:
                                subsection_links.append({
                                    'section_id': section_text,
                                    'title': title_text,
                                    'url': href
                                })
            
            # Strategy 2: Find direct links containing section identifiers
            if not subsection_links:
                all_page_links = driver.find_elements(By.TAG_NAME, "a")
                for link in all_page_links:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    
                    # Look for patterns like AP7.C1.1, AP7.C2.3, etc.
                    if href and text and ("AP7.C" in text or "ap7-c" in href.lower()):
                        # Extract section ID from text
                        parts = text.split()
                        section_id = parts[0] if parts else text[:20]
                        title = ' '.join(parts[1:]) if len(parts) > 1 else text
                        
                        subsection_links.append({
                            'section_id': section_id,
                            'title': title,
                            'url': href
                        })
        except Exception as e:
            print(f"    ✗ Error finding subsections: {e}")
        
        # Remove duplicates from subsection links
        seen_subsection_urls = set()
        unique_subsections = []
        for sub in subsection_links:
            sub_url = urljoin(base_url, sub['url'])
            if sub_url not in seen_subsection_urls and sub_url not in visited_urls:
                seen_subsection_urls.add(sub_url)
                unique_subsections.append(sub)
        
        if unique_subsections:
            page_data['has_subsections'] = True
            print(f"    ✓ Found {len(unique_subsections)} subsections")
            
            # Add subsections to queue for processing
            for sub in unique_subsections:
                queue.append({
                    'url': urljoin(base_url, sub['url']),
                    'level': current['level'] + 1,
                    'parent': current['section_id'],
                    'section_id': sub['section_id'],
                    'title': sub['title']
                })
        else:
            print(f"    ✓ No subsections (leaf node)")
        
        # Add current page to results
        all_links.append(page_data)
        
        # Progress indicator
        print(f"    Total pages discovered: {len(all_links)}, Queue: {len(queue)}")
    
    print(f"\n[4/6] Discovery complete! Found {len(all_links)} total pages")
    
    # Convert to DataFrame
    print("[5/6] Creating comprehensive CSV...")
    df = pd.DataFrame(all_links)
    
    # Sort by section_id for better organization
    df = df.sort_values('section_id')
    
    # Save to CSV
    output_file = "dsca_complete_hierarchy.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"     ✓ Data saved to: {output_file}")
    
    # Create summary report
    print("\n[6/6] Generating summary report...")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PARSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total pages parsed: {len(df)}")
    print(f"\nPages by level:")
    for level in sorted(df['level'].unique()):
        count = len(df[df['level'] == level])
        print(f"  Level {level}: {count} pages")
    
    print(f"\nLeaf nodes (no subsections): {len(df[df['has_subsections'] == False])}")
    
    # Show sample of deepest level
    max_level = df['level'].max()
    deepest = df[df['level'] == max_level].head(5)
    print(f"\nSample of deepest level pages (Level {max_level}):")
    for idx, row in deepest.iterrows():
        print(f"  - {row['section_id']}: {row['title'][:60]}")
    
    # Save detailed report
    report_file = "dsca_parsing_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("DSCA Deep Link Parsing Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total pages parsed: {len(df)}\n")
        f.write(f"Maximum depth: {max_level} levels\n\n")
        f.write("All discovered pages:\n")
        f.write("-"*60 + "\n")
        for idx, row in df.iterrows():
            indent = "  " * row['level']
            f.write(f"{indent}{row['section_id']}: {row['title']}\n")
            f.write(f"{indent}  URL: {row['url']}\n\n")
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    print(traceback.format_exc())

finally:
    print("\nClosing browser...")
    driver.quit()
    print("Done!")


"""
HOW THIS WORKS:
================

1. INITIALIZATION
   - Starts at the main Appendix 7 page
   - Uses a BFS (Breadth-First Search) queue to process pages

2. RECURSIVE DISCOVERY (BFS)
   - For each page:
     a) Visit the page
     b) Extract content preview
     c) Find all subsection links (using multiple strategies)
     d) Add subsections to queue for processing
   - Continues until no more links to process

3. LINK DISCOVERY STRATEGIES
   Strategy 1: Parse tables with section listings
   - Common pattern: Section ID in col 1, Title in col 2
   
   Strategy 2: Find direct links with section patterns
   - Looks for AP7.C patterns in URLs and text

4. OUTPUT
   - CSV with complete hierarchy
   - Text report with indented structure
   - Statistics by level

5. KEY FEATURES
   - Avoids duplicates (tracks visited URLs)
   - Records parent-child relationships
   - Tracks depth level for each page
   - Extracts content preview from each page
   - Handles different page structures
"""