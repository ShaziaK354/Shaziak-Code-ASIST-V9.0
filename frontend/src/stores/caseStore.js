import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { useAuthStore } from './authStore'; 

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// LocalStorage Helper Functions
const LS_KEYS = {
    PINNED_CASES: 'asist_vue_pinnedCases',
    RECENT_CASES: 'asist_vue_recentCases',
    ACTIVE_CASE_DATA: 'asist_vue_lastActiveCaseData', 
    EXPANDED_COUNTRIES: 'asist_vue_expandedCountries'
};

function saveToLS(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.error("Error saving to localStorage:", e);
    }
}

function loadFromLS(key, defaultValue = null) {
    try {
        const storedValue = localStorage.getItem(key);
        return storedValue ? JSON.parse(storedValue) : defaultValue;
    } catch (e) {
        console.error("Error loading from localStorage:", e);
        return defaultValue;
    }
}

function getCountryNameFromCode(code) {
    if (!code) return 'Unknown Country';
    const map = { 
        jp: 'Japan', kr: 'South Korea', au: 'Australia', mx: 'Mexico', 
        jo: 'Jordan', eg: 'Egypt', tw: 'Taiwan', id: 'Indonesia', 
        pl: 'Poland', gr: 'Greece', gb: 'United Kingdom', sg: 'Singapore', 
        ae: 'United Arab Emirates', sa: 'Saudi Arabia', qa: 'Qatar', 
        no: 'Norway', il: 'Israel', tr: 'Turkey', br: 'Brazil', 
        nl: 'Netherlands', in: 'India', ca: 'Canada', th: 'Thailand', 
        ma: 'Morocco', it: 'Italy', un: 'Unknown' 
    };
    return map[code.toLowerCase()] || code.toUpperCase();
}


export const useCaseStore = defineStore('cases', () => {
  // --- State ---
  const allCases = ref([]); 
  const activeCaseDetails = ref(loadFromLS(LS_KEYS.ACTIVE_CASE_DATA)); 
  const isLoadingCases = ref(false);
  const caseError = ref(null);
  const isUploadingDocument = ref(false); 
  const documentUploadError = ref(null);  
  const isDeletingCaseDocument = ref(false);
  const caseDocumentDeleteError = ref(null);

  const pinnedCaseIds = ref(loadFromLS(LS_KEYS.PINNED_CASES, []));
  const recentCasesData = ref(loadFromLS(LS_KEYS.RECENT_CASES, [])); 
  const MAX_RECENT_CASES = 5;
  const expandedCountriesInSidebar = ref(loadFromLS(LS_KEYS.EXPANDED_COUNTRIES, []));


  // --- Getters ---
  const casesByCountry = computed(() => {
    const grouped = {};
    allCases.value.forEach(caseItem => {
      const countryCode = (caseItem.country || 'un').toLowerCase(); 
      const countryName = getCountryNameFromCode(countryCode);
      
      if (!grouped[countryCode]) { 
        grouped[countryCode] = {
          name: countryName,
          code: countryCode, 
          cases: []
        };
      }
      const displayCase = {
        id: caseItem.id || caseItem.caseNumber,
        name: caseItem.caseDescription || `Case ${caseItem.id || caseItem.caseNumber}`,
        iconText: (caseItem.caseNumber || caseItem.id || 'N/A').slice(-3).toUpperCase(),
        ...caseItem 
      };
      grouped[countryCode].cases.push(displayCase);
    });
    return Object.values(grouped).sort((a, b) => a.name.localeCompare(b.name));
  });

  const getCaseById = computed(() => {
    return (caseId) => allCases.value.find(c => (c.id === caseId || c.caseNumber === caseId));
  });

  const renderedPinnedCases = computed(() => {
    return pinnedCaseIds.value.map(caseId => {
        const caseDetail = getCaseById.value(caseId);
        if (caseDetail) {
            return { 
                id: caseDetail.id || caseDetail.caseNumber, 
                name: caseDetail.caseDescription || `Case ${caseId}`, 
                iconText: (caseDetail.caseNumber || caseDetail.id || 'N/A').slice(-3).toUpperCase(),
                countryCode: caseDetail.country 
            };
        }
        const recentMatch = recentCasesData.value.find(rc => rc.id === caseId);
        if (recentMatch) return recentMatch;
        return { id: caseId, name: `Pinned: ${caseId}`, iconText: caseId.slice(-3).toUpperCase(), countryCode: 'un' };
    }).filter(Boolean);
  });

  const isCasePinned = computed(() => {
    return (caseId) => pinnedCaseIds.value.includes(caseId);
  });


  // --- Actions ---
  async function fetchUserCases() {
    const authStore = useAuthStore();
    if (!authStore.isLoggedIn || !authStore.userProfile?.sub) { 
      allCases.value = []; 
      activeCaseDetails.value = null; 
      localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
      return;
    }
    isLoadingCases.value = true;
    caseError.value = null;
    try {
      // ✅ ALWAYS clear localStorage to force SR-P-NAV as default
      localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
      
      const response = await fetch('/api/user/cases'); 
      if (!response.ok) {
        if (response.status === 401) {
            authStore.user = null; 
            allCases.value = [];
            activeCaseDetails.value = null;
            localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
            throw new Error('User not authenticated. Please log in again.');
        }
        const errorData = await response.json().catch(() => ({ message: `Failed to fetch cases: ${response.statusText}` }));
        throw new Error(errorData.message || `Failed to fetch cases: ${response.status}`);
      }
      const data = await response.json();
      // Ensure each case object has a caseDocuments array for reactivity, even if empty
      allCases.value = data.map(c => ({ ...c, caseDocuments: Array.isArray(c.caseDocuments) ? c.caseDocuments : [] }));
      console.log('[CaseStore] Cases fetched and processed:', JSON.parse(JSON.stringify(allCases.value)));

      // ✅ ALWAYS default to SR-P-NAV if it exists
      let caseToMakeActive = null;
      const srPNavCase = allCases.value.find(c => c.caseNumber === 'SR-P-NAV');
      
      if (srPNavCase) {
        caseToMakeActive = srPNavCase;
        console.log('[CaseStore] ✅ Defaulting to SR-P-NAV');
      } else if (allCases.value.length > 0) {
        // Fallback to first case if SR-P-NAV doesn't exist
        caseToMakeActive = allCases.value[0];
        console.log('[CaseStore] ⚠️ SR-P-NAV not found, using first case:', caseToMakeActive.caseNumber);
      }
      
      if (caseToMakeActive) {
        // Call setActiveCase with the full object from allCases to ensure reactivity
        await setActiveCase(caseToMakeActive.id || caseToMakeActive.caseNumber, caseToMakeActive);
      } else {
        activeCaseDetails.value = null; 
        localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
      }

    } catch (e) {
      console.error('[CaseStore] Error fetching cases:', e);
      caseError.value = e.message;
      allCases.value = []; 
      activeCaseDetails.value = null;
      localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
    } finally {
      isLoadingCases.value = false;
    }
  }

  async function fetchCaseDetails(caseId) {
    if (!caseId) {
      console.error('[CaseStore] fetchCaseDetails: No caseId provided');
      return null;
    }

    const authStore = useAuthStore();
    if (!authStore.isLoggedIn) {
      console.warn('[CaseStore] fetchCaseDetails: User not authenticated');
      return null;
    }

    const wasCaseLoading = isLoadingCases.value;
    if (!wasCaseLoading) {
      isLoadingCases.value = true;
    }
    
    caseError.value = null;

    try {
      console.log(`[CaseStore] Fetching case details for: ${caseId}`);
      
      const response = await fetch(`/api/cases/${encodeURIComponent(caseId)}`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          authStore.user = null;
          throw new Error('User not authenticated. Please log in again.');
        }
        
        const errorData = await response.json().catch(() => ({ 
          error: `Failed to fetch case: ${response.statusText}` 
        }));
        throw new Error(errorData.error || `Failed to fetch case: ${response.status}`);
      }

      const caseData = await response.json();
      
      console.log(`[CaseStore] ✅ Fetched case: ${caseData.caseNumber || caseData.id}`);
      console.log(`[CaseStore]   Documents: ${caseData.caseDocuments?.length || 0}`);
      
      caseData.caseDocuments = Array.isArray(caseData.caseDocuments) 
        ? caseData.caseDocuments 
        : [];
      
      const financialDocs = caseData.caseDocuments.filter(doc => 
        doc.documentType === 'FINANCIAL_DATA' || doc.document_type === 'FINANCIAL_DATA'
      );
      
      if (financialDocs.length > 0) {
        console.log(`[CaseStore]   📊 Financial documents: ${financialDocs.length}`);
        financialDocs.forEach(doc => {
          const records = doc.metadata?.financialRecords || doc.financial_records || [];
          const filename = doc.filename || doc.fileName;
          console.log(`[CaseStore]     • ${filename}: ${records.length} records`);
        });
      }

      const existingIndex = allCases.value.findIndex(c => 
        c.id === caseData.id || c.caseNumber === caseData.caseNumber
      );

      if (existingIndex !== -1) {
        allCases.value[existingIndex] = { ...caseData };
        console.log(`[CaseStore] ✅ Updated case in allCases array`);
      } else {
        allCases.value.push({ ...caseData });
        console.log(`[CaseStore] ✅ Added new case to allCases array`);
      }

      if (activeCaseDetails.value && 
          (activeCaseDetails.value.id === caseData.id || 
           activeCaseDetails.value.caseNumber === caseData.caseNumber)) {
        
        activeCaseDetails.value = {
          ...caseData,
          caseDocuments: [...caseData.caseDocuments.map(doc => ({...doc}))]
        };
        
        saveToLS(LS_KEYS.ACTIVE_CASE_DATA, activeCaseDetails.value);
        console.log(`[CaseStore] ✅ Updated activeCaseDetails for active case`);
        console.log(`[CaseStore]   Current caseDocuments:`, 
          JSON.parse(JSON.stringify(activeCaseDetails.value.caseDocuments)));
      }

      return caseData;

    } catch (error) {
      console.error(`[CaseStore] ❌ Error fetching case ${caseId}:`, error);
      caseError.value = error.message;
      return null;
    } finally {
      if (!wasCaseLoading) {
        isLoadingCases.value = false;
      }
    }
  }

  async function setActiveCase(caseId, preFetchedCaseData = null) {
    const caseData = preFetchedCaseData || getCaseById.value(caseId);
    if (caseData) {
      const currentDocs = Array.isArray(caseData.caseDocuments) ? [...caseData.caseDocuments.map(doc => ({...doc}))] : [];
      activeCaseDetails.value = { ...caseData, caseDocuments: currentDocs }; 
      
      saveToLS(LS_KEYS.ACTIVE_CASE_DATA, activeCaseDetails.value);
      addRecentCase(activeCaseDetails.value);
      console.log('[CaseStore] Active case set/updated:', JSON.parse(JSON.stringify(activeCaseDetails.value)));
    } else {
      console.warn(`[CaseStore] Case with ID ${caseId} not found in allCases. Cannot set as active.`);
      activeCaseDetails.value = null;
      localStorage.removeItem(LS_KEYS.ACTIVE_CASE_DATA);
    }
  }

  function togglePinCase(caseId) {
    const index = pinnedCaseIds.value.indexOf(caseId);
    if (index > -1) {
      pinnedCaseIds.value.splice(index, 1);
    } else {
      pinnedCaseIds.value.unshift(caseId);
    }
    saveToLS(LS_KEYS.PINNED_CASES, pinnedCaseIds.value);
  }

  function addRecentCase(caseDataObj) {
    if (!caseDataObj || !(caseDataObj.id || caseDataObj.caseNumber)) return;
    const id = caseDataObj.id || caseDataObj.caseNumber;
    const caseToAdd = { 
        id: id,
        name: caseDataObj.caseDescription || caseDataObj.name || `Case ${id}`,
        iconText: (caseDataObj.caseNumber || id || 'N/A').slice(-3).toUpperCase(),
        countryCode: caseDataObj.country 
    };
    recentCasesData.value = recentCasesData.value.filter(c => c.id !== caseToAdd.id);
    recentCasesData.value.unshift(caseToAdd);
    if (recentCasesData.value.length > MAX_RECENT_CASES) {
      recentCasesData.value.pop();
    }
    saveToLS(LS_KEYS.RECENT_CASES, recentCasesData.value);
  }
  
  function saveExpandedCountries(countriesArray) {
    expandedCountriesInSidebar.value = countriesArray;
    saveToLS(LS_KEYS.EXPANDED_COUNTRIES, expandedCountriesInSidebar.value);
  }

  async function uploadCaseDocument(caseId, filesToUpload) {
    const authStore = useAuthStore();
    if (!authStore.isLoggedIn) {
      documentUploadError.value = "User not authenticated. Please log in.";
      return null;
    }
    if (!caseId || !filesToUpload || filesToUpload.length === 0) {
      documentUploadError.value = "No case ID or files provided for upload.";
      return null;
    }

    isUploadingDocument.value = true;
    documentUploadError.value = null;
    let apiResponse;

    const formData = new FormData();
    for (const file of filesToUpload) {
      formData.append('documents', file, file.name);
    }

    try {
      apiResponse = await fetch(`/api/cases/${caseId}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        let errorMsg = `Failed to upload documents: ${apiResponse.status} ${apiResponse.statusText}`;
        try {
          const errorData = await apiResponse.json();
          errorMsg = errorData.error || errorData.message || errorMsg;
        } catch (jsonError) {
          console.warn("[CaseStore] Response body was not JSON for error during upload:", await apiResponse.text().catch(()=>"Could not read error response body"));
        }
        throw new Error(errorMsg);
      }

      const responseData = await apiResponse.json();
      console.log('[CaseStore] Documents uploaded successfully (raw response):', responseData);
      
      if (activeCaseDetails.value && (activeCaseDetails.value.id === caseId || activeCaseDetails.value.caseNumber === caseId)) {
        const existingDocs = Array.isArray(activeCaseDetails.value.caseDocuments) ? [...activeCaseDetails.value.caseDocuments] : [];
        
        let newDocsMetadata = [];
        if (responseData.uploadedDocuments && Array.isArray(responseData.uploadedDocuments)) {
          newDocsMetadata = responseData.uploadedDocuments.map(doc => ({ ...doc }));
        }
        
        const updatedDocs = [...existingDocs, ...newDocsMetadata];
        
        activeCaseDetails.value = {
          ...activeCaseDetails.value,
          caseDocuments: updatedDocs
        };
        
        console.log('[CaseStore] Updated activeCaseDetails.caseDocuments:', JSON.parse(JSON.stringify(activeCaseDetails.value.caseDocuments)));
        saveToLS(LS_KEYS.ACTIVE_CASE_DATA, activeCaseDetails.value);
      }
      return responseData.uploadedDocuments || [];

    } catch (e) {
      console.error('[CaseStore] Error during document upload process:', e);
      documentUploadError.value = e.message;
      return null;
    } finally {
      isUploadingDocument.value = false;
    }
  }

  async function deleteCaseDocument(caseId, documentIdentifier) {
    const authStore = useAuthStore();
    if (!authStore.isLoggedIn) {
      caseDocumentDeleteError.value = "User not authenticated.";
      return;
    }

    if (!caseId || !documentIdentifier) {
      caseDocumentDeleteError.value = "Case ID or Document Identifier missing for deletion.";
      console.error("[CaseStore] Missing caseId or documentIdentifier for deletion.", { caseId, documentIdentifier });
      return;
    }

    isDeletingCaseDocument.value = true;
    caseDocumentDeleteError.value = null;

    try {
      const response = await fetch(`${API_BASE_URL}/cases/documents/delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authStore.token}`,
        },
        body: JSON.stringify({
          caseId: caseId,
          documentId: documentIdentifier
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: `Failed to delete document. Status: ${response.status}` }));
        throw new Error(errorData.message || `API error: ${response.status}`);
      }

      if (activeCaseDetails.value && activeCaseDetails.value.id === caseId && activeCaseDetails.value.caseDocuments) {
        activeCaseDetails.value.caseDocuments = activeCaseDetails.value.caseDocuments.filter(
          doc => (doc.documentId || doc.fileName) !== documentIdentifier
        );
      }

    } catch (e) {
      console.error('[CaseStore] Error deleting case document:', e);
      caseDocumentDeleteError.value = e.message;
    } finally {
      isDeletingCaseDocument.value = false;
    }
  }


  return {
    allCases, activeCaseDetails, isLoadingCases, caseError,
    isUploadingDocument, documentUploadError, 
    pinnedCaseIds, recentCasesData, expandedCountriesInSidebar,
    casesByCountry, getCaseById, renderedPinnedCases, isCasePinned,
    fetchUserCases, 
    fetchCaseDetails,
    setActiveCase, togglePinCase, addRecentCase,
    saveExpandedCountries, uploadCaseDocument, 
    isDeletingCaseDocument,
    caseDocumentDeleteError,
    deleteCaseDocument,
  };
});