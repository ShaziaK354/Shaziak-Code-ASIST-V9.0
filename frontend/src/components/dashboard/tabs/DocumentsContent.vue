<template>
  <div class="tab-content-section-wrapper documents-content-wrapper">
    <div class="section-title">
      <h3>Document Management</h3>
    </div>
    <div class="documents-container">
      <nav class="document-sub-tabs">
        <div 
          class="document-sub-tab" 
          :class="{ active: activeSubTab === 'case' }"
          @click="setActiveSubTab('case')"
          role="tab"
          :aria-selected="activeSubTab === 'case'"
        >
          Case Documents
        </div>
        <div 
          class="document-sub-tab" 
          :class="{ active: activeSubTab === 'chat' }"
          @click="setActiveSubTab('chat')"
          role="tab"
          :aria-selected="activeSubTab === 'chat'"
        >
          Chat Session Context
        </div>
      </nav>
      
      <div v-if="activeSubTab === 'case'" class="document-items-container" id="caseDocumentsList">
        <div 
          v-for="doc in displayedCaseDocuments" 
          :key="doc.documentId || doc.fileName" 
          class="document-item" 
          :title="doc.fileName"
          @mouseenter="hoveredDocument = doc.documentId || doc.fileName"
          @mouseleave="hoveredDocument = null"
        >
          <div class="document-icon"><i :class="getFileFAIconClass(doc.fileName)"></i></div>
          <div class="document-info">
            <div class="document-title">{{ doc.fileName }}</div>
            <div class="document-meta">
              {{ doc.fileType?.split('/').pop()?.toUpperCase() || 'FILE' }} &bull; 
              {{ formatFileSize(doc.sizeBytes) }} &bull; 
              Uploaded: {{ formatDate(doc.uploadedAt) }}
            </div>
          </div>
          <div class="document-actions" v-show="hoveredDocument === (doc.documentId || doc.fileName)">
            <a :href="doc.url" target="_blank" class="document-action-btn download" title="Download"><i class="fas fa-download"></i></a>
            <button class="document-action-btn delete" title="Delete" @click="deleteCaseDocument(doc)"><i class="fas fa-trash-alt"></i></button>
          </div>
        </div>
        <div v-if="displayedCaseDocuments.length === 0 && !caseStore.isUploadingDocument && !caseStore.isLoadingCases" class="empty-documents-message">
            No case documents uploaded yet for this case.
        </div>
        <div v-if="caseStore.isUploadingDocument" class="upload-status-message">
            <i class="fas fa-spinner fa-spin"></i> Uploading case document(s)...
        </div>
        <div v-if="caseStore.isDeletingCaseDocument" class="upload-status-message">
            <i class="fas fa-spinner fa-spin"></i> Deleting case document...
        </div>
         <div v-if="caseStore.documentUploadError" class="upload-status-message error">
            Upload Error: {{ caseStore.documentUploadError }}
        </div>
        <div v-if="caseStore.caseDocumentDeleteError" class="upload-status-message error">
            Delete Error: {{ caseStore.caseDocumentDeleteError }}
        </div>

        <div class="file-upload-area" id="caseUploadArea" 
             @dragenter.prevent.stop="handleDragEnter($event, 'case')"
             @dragover.prevent.stop="handleDragOver($event, 'case')"
             @dragleave.prevent.stop="handleDragLeave($event, 'case')"
             @drop.prevent.stop="handleDrop($event, 'case')">
          <p>Drag & drop files here or click to upload for this case</p>
          <button class="btn btn-outline" @click="triggerFileInput(caseFileInputRef)">
            <span class="nav-icon"><i class="fas fa-upload"></i></span> Select Files
          </button>
          <input type="file" id="caseFileInputVue" ref="caseFileInputRef" @change="handleFileSelect($event, 'case')" multiple style="display: none;">
        </div>
      </div>

      <div v-if="activeSubTab === 'chat'" class="document-items-container" id="chatContextDocumentsList">
        <div 
          v-for="doc in chatStore.currentSessionContextDocuments" 
          :key="doc.documentId || doc.fileName" 
          class="document-item" 
          :title="doc.fileName"
          @mouseenter="hoveredDocument = doc.documentId || doc.fileName"
          @mouseleave="hoveredDocument = null"
        >
          <div class="document-icon"><i :class="getFileFAIconClass(doc.fileName)"></i></div>
          <div class="document-info">
            <div class="document-title">{{ doc.fileName }}</div>
            <div class="document-meta">
                {{ doc.fileType?.split('/').pop()?.toUpperCase() || 'FILE' }} &bull; 
                {{ formatFileSize(doc.sizeBytes) }} &bull; 
                Staged: {{ formatDate(doc.uploadedAt) }}
            </div>
          </div>
           <div class="document-actions" v-show="hoveredDocument === (doc.documentId || doc.fileName)">
            <a :href="doc.url" target="_blank" class="document-action-btn download" title="Download"><i class="fas fa-download"></i></a>
            <button class="document-action-btn delete" title="Remove from session context" @click="removeStagedChatDocument(doc.documentId)"><i class="fas fa-times-circle"></i></button>
          </div>
        </div>
         <div v-if="chatStore.currentSessionContextDocuments.length === 0 && !chatStore.isUploadingToSession" class="empty-documents-message">
            No documents added to this chat session's context yet.
        </div>
        <div v-if="chatStore.isUploadingToSession" class="upload-status-message">
            <i class="fas fa-spinner fa-spin"></i> Adding file to session context...
        </div>
         <div v-if="chatStore.fileUploadError" class="upload-status-message error"> Error adding file: {{ chatStore.fileUploadError }}
        </div>
        <div class="file-upload-area" id="chatContextUploadArea" 
             @dragenter.prevent.stop="handleDragEnter($event, 'chatContext')"
             @dragover.prevent.stop="handleDragOver($event, 'chatContext')"
             @dragleave.prevent.stop="handleDragLeave($event, 'chatContext')"
             @drop.prevent.stop="handleDrop($event, 'chatContext')">
          <p>Drag & drop files here or click to add to current chat session context</p>
          <button class="btn btn-outline" @click="triggerFileInput(chatContextFileInputRef)">
            <span class="nav-icon"><i class="fas fa-plus-circle"></i></span> Add Files to Session
          </button>
          <input type="file" id="chatContextFileInputVue" ref="chatContextFileInputRef" @change="handleFileSelect($event, 'chatContext')" multiple style="display: none;">
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue';
import { useCaseStore } from '@/stores/caseStore'; 
import { useChatStore } from '@/stores/chatStore'; 

const props = defineProps({
  activeCaseId: { 
    type: String,
    default: null
  },
  activeCaseData: { 
    type: Object,
    default: () => null
  }
});

const caseStore = useCaseStore();
const chatStore = useChatStore(); 

const activeSubTab = ref('case'); 
const hoveredDocument = ref(null);
const caseFileInputRef = ref(null); 
const chatContextFileInputRef = ref(null);
const forceRefreshKey = ref(0);

const setActiveSubTab = (tabName) => {
  activeSubTab.value = tabName;
};

// ✅ ULTIMATE FIX: Check ALL possible locations for documents
const displayedCaseDocuments = computed(() => {
  // Trigger reactivity
  forceRefreshKey.value;
  
  const caseId = props.activeCaseId;
  let foundCase = null;
  
  // Strategy 1: Check activeCaseDetails
  let docs = caseStore.activeCaseDetails?.caseDocuments || 
             caseStore.activeCaseDetails?.documents;
  
  // Strategy 2: Check props
  if (!docs || docs.length === 0) {
    docs = props.activeCaseData?.caseDocuments || 
           props.activeCaseData?.documents;
  }
  
  // Strategy 3: Check activeCase
  if (!docs || docs.length === 0) {
    docs = caseStore.activeCase?.caseDocuments || 
           caseStore.activeCase?.documents;
  }
  
  // Strategy 4: Search in allCases array by case ID
  if ((!docs || docs.length === 0) && caseId) {
    foundCase = caseStore.allCases?.find(c => 
      c.id === caseId || 
      c.caseNumber === caseId || 
      c.caseId === caseId
    );
    docs = foundCase?.caseDocuments || foundCase?.documents;
  }
  
  const finalDocs = docs || [];
  
  console.log('[DocumentsContent] Computing documents:', {
    caseId: caseId,
    storeDetailsCaseDocuments: caseStore.activeCaseDetails?.caseDocuments?.length || 0,
    storeDetailsDocuments: caseStore.activeCaseDetails?.documents?.length || 0,
    propsCaseDocuments: props.activeCaseData?.caseDocuments?.length || 0,
    propsDocuments: props.activeCaseData?.documents?.length || 0,
    storeCaseDocuments: caseStore.activeCase?.caseDocuments?.length || 0,
    storeDocuments: caseStore.activeCase?.documents?.length || 0,
    allCasesCount: caseStore.allCases?.length || 0,
    foundInAllCases: foundCase ? `Yes (${(foundCase.caseDocuments || foundCase.documents || []).length} docs)` : 'No',
    totalFound: finalDocs.length,
    forceRefreshKey: forceRefreshKey.value
  });
  
  return finalDocs;
});

// ✅ Watch activeCaseDetails - ONLY log, don't trigger refresh (handled by activeCaseId watcher)
watch(() => caseStore.activeCaseDetails, (newDetails) => {
  if (newDetails) {
    console.log('[DocumentsContent] Store activeCaseDetails changed:', {
      caseId: newDetails?.id,
      caseNumber: newDetails?.caseNumber,
      documentsCount: (newDetails?.documents || newDetails?.caseDocuments || []).length,
      hasDocuments: !!(newDetails?.documents || newDetails?.caseDocuments)
    });
    // Removed: forceRefreshKey.value++ - this was causing infinite loop
  }
}, { deep: false }); // Changed: removed immediate, deep:false to prevent over-triggering

// ✅ Watch activeCase - ONLY log
watch(() => caseStore.activeCase, (newCase) => {
  if (newCase) {
    console.log('[DocumentsContent] Store activeCase changed:', {
      caseId: newCase?.id,
      documentsCount: (newCase?.documents || newCase?.caseDocuments || []).length
    });
    // Removed: forceRefreshKey.value++ - this was causing infinite loop
  }
}, { deep: false }); // Changed: removed immediate, deep:false

// ✅ Watch allCases - ONLY when documents actually change for THIS case
let lastDocCount = 0;
watch(() => caseStore.allCases, (newCases) => {
  if (newCases && props.activeCaseId) {
    const foundCase = newCases.find(c => 
      c.id === props.activeCaseId || 
      c.caseNumber === props.activeCaseId || 
      c.caseId === props.activeCaseId
    );
    if (foundCase) {
      const docCount = (foundCase.caseDocuments || foundCase.documents || []).length;
      console.log('[DocumentsContent] Found case in allCases:', {
        caseId: foundCase.id,
        caseNumber: foundCase.caseNumber,
        documentsCount: docCount
      });
      // Only refresh if document count actually changed
      if (docCount !== lastDocCount) {
        lastDocCount = docCount;
        forceRefreshKey.value++;
      }
    }
  }
}, { deep: false }); // Changed: removed immediate, deep:false

// ✅ Watch case data changes from props - ONLY on actual data change
watch(() => props.activeCaseData?.caseDocuments?.length || props.activeCaseData?.documents?.length || 0, (newCount, oldCount) => {
  if (newCount !== oldCount) {
    console.log('[DocumentsContent] Props document count changed:', { newCount, oldCount });
    forceRefreshKey.value++;
  }
});

const getFileFAIconClass = (filename) => {
  if (!filename) return 'fas fa-file';
  const ext = filename.split('.').pop().toLowerCase();
  if (['pdf'].includes(ext)) return 'fas fa-file-pdf';
  if (['doc', 'docx'].includes(ext)) return 'fas fa-file-word';
  if (['xls', 'xlsx'].includes(ext)) return 'fas fa-file-excel';
  return 'fas fa-file';
};

const formatFileSize = (bytes) => {
  if (bytes === null || bytes === undefined || isNaN(Number(bytes))) return 'N/A';
  if (Number(bytes) === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(Number(bytes)) / Math.log(k));
  return parseFloat((Number(bytes) / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
    } catch (e) { 
        console.warn(`[DocumentsContent] Error formatting date '${dateString}':`, e.message); 
        return 'Invalid Date';
    }
};

const triggerFileInput = (refInstance) => {
  if (refInstance) {
    refInstance.click();
  }
};

const processAndUploadFiles = async (files, targetType) => {
  if (targetType === 'case') {
    const caseId = props.activeCaseData?.id || props.activeCaseId;
    
    if (!caseId) {
      alert("No active case selected to upload documents to.");
      caseStore.documentUploadError = "No active case selected."; 
      return;
    }
    
    console.log('[DocumentsContent] Uploading to case:', caseId);
    
    await caseStore.uploadCaseDocument(caseId, Array.from(files));
    
    if (!caseStore.documentUploadError) {
      console.log('[DocumentsContent] ✅ Upload successful, refreshing...');
      
      await nextTick();
      await caseStore.fetchCaseDetails(caseId);
      await nextTick();
      
      forceRefreshKey.value++;
      
      console.log('[DocumentsContent] 🔄 After refresh, documents:', displayedCaseDocuments.value.length);
      
      alert(`✅ ${files.length} document(s) uploaded successfully!`);
    } else {
      console.error('[DocumentsContent] ❌ Upload error:', caseStore.documentUploadError);
      alert(`❌ Upload failed: ${caseStore.documentUploadError}`);
    }
    
  } else if (targetType === 'chatContext') {
    console.log('[DocumentsContent] Adding to chat context...');
    for (const file of Array.from(files)) {
        await chatStore.addFileToSessionContext(file);
    }
  }
};

const handleFileSelect = (event, targetType) => {
  const files = event.target.files;
  if (files && files.length > 0) {
    processAndUploadFiles(files, targetType);
  }
  if (event.target) event.target.value = ''; 
};

const handleDragEnter = (event, targetType) => { 
  event.currentTarget.classList.add('dragover'); 
};

const handleDragOver = (event, targetType) => { 
  event.preventDefault(); 
  event.currentTarget.classList.add('dragover');
};

const handleDragLeave = (event, targetType) => { 
  event.currentTarget.classList.remove('dragover');
};

const handleDrop = (event, targetType) => {
  event.currentTarget.classList.remove('dragover');
  const files = event.dataTransfer.files;
  if (files && files.length > 0) {
    processAndUploadFiles(files, targetType);
  }
};

const deleteCaseDocument = async (documentItem) => {
    const caseId = props.activeCaseData?.id || props.activeCaseId;
    
    if (!caseId) {
        alert("No active case selected.");
        return;
    }
    
    const docId = documentItem.documentId;
    if (!docId) {
        alert("Document ID missing, cannot delete.");
        return;
    }
    
    if (confirm(`Are you sure you want to permanently delete "${documentItem.fileName}" from this case?`)) {
        console.log(`[DocumentsContent] Requesting deletion of case doc ID: ${docId} from case: ${caseId}`);
        
        await caseStore.deleteCaseDocument(caseId, docId);
        
        if (!caseStore.caseDocumentDeleteError) {
          console.log('[DocumentsContent] Delete successful, refreshing case details...');
          
          await nextTick();
          await caseStore.fetchCaseDetails(caseId);
          await nextTick();
          
          forceRefreshKey.value++;
          
          alert('✅ Document deleted successfully!');
        } else {
          alert(`❌ Delete failed: ${caseStore.caseDocumentDeleteError}`);
        }
    }
};

const removeStagedChatDocument = async (documentId) => {
    if (confirm(`Are you sure you want to remove this document from the current chat session's context?`)) {
        console.log(`[DocumentsContent] Removing chat context document ID: ${documentId}`);
        await chatStore.removeFileFromSessionContext(documentId);
    }
};

watch(() => props.activeCaseId, async (newCaseId, oldCaseId) => {
  if (newCaseId && newCaseId !== oldCaseId) {
    console.log(`[DocumentsContent] Active case changed from ${oldCaseId} to ${newCaseId}`);
    activeSubTab.value = 'case';
    
    await nextTick();
    await caseStore.fetchCaseDetails(newCaseId);
    await nextTick();
    forceRefreshKey.value++;
  }
}); // Removed immediate: true - this was causing initial double-fetch

onMounted(() => {
    console.log('[DocumentsContent] Component mounted', {
      activeCaseId: props.activeCaseId,
      documentCount: displayedCaseDocuments.value.length,
      storeHasDetails: !!caseStore.activeCaseDetails,
      storeHasCase: !!caseStore.activeCase,
      allCasesCount: caseStore.allCases?.length || 0
    });
});

</script>

<style scoped>
.tab-content-section-wrapper {
  padding: var(--space-sm);
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden; 
}
.section-title {
  margin-bottom: var(--space-sm);
  padding-bottom: var(--space-xs);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.section-title h3 {
  font-size: 1rem;
  color: var(--primary);
  margin: 0;
}
.documents-container {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.document-sub-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--space-sm);
  flex-shrink: 0;
}
.document-sub-tab {
  padding: var(--space-xs) var(--space-md);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-size: 0.85rem;
  color: #666;
  font-weight: 500;
  transition: color 0.2s ease, border-bottom-color 0.2s ease;
}
.document-sub-tab.active {
  border-bottom-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}
.document-sub-tab:hover {
    color: var(--accent);
}
.document-items-container {
  flex-grow: 1;
  overflow-y: auto; 
}
.empty-documents-message {
    padding: var(--space-md);
    text-align: center;
    color: #777;
}
.upload-status-message {
    padding: var(--space-sm) var(--space-md);
    text-align: center;
    color: var(--accent);
    font-style: italic;
}
.upload-status-message.error {
    color: var(--danger);
    font-style: normal;
    font-weight: bold;
}
.upload-status-message i {
    margin-right: var(--space-xs);
}
.document-item {
  display: flex;
  align-items: center;
  padding: var(--space-sm); 
  border-radius: 4px;
  transition: background-color 0.2s;
  cursor: default; 
  margin-bottom: var(--space-xs);
  position: relative; 
  border: 1px solid transparent; 
}
.document-item:hover {
  background-color: #f5f7fa; 
  border-color: var(--border);
}
.document-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: var(--space-sm);
  flex-shrink: 0;
  font-size: 1.5rem; 
  color: var(--secondary);
}
.document-info {
  flex: 1;
  overflow: hidden; 
}
.document-title {
  font-weight: 500;
  font-size: 0.9rem;
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text-dark);
}
.document-meta {
  font-size: 0.75rem;
  color: #666;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.document-actions {
  display: flex; 
  gap: var(--space-xs);
  opacity: 0; 
  transition: opacity 0.2s ease-in-out;
}
.document-item:hover .document-actions {
  opacity: 1; 
}
.document-action-btn {
  background: none;
  border: none;
  padding: var(--space-xs);
  cursor: pointer;
  color: #666;
  font-size: 0.9rem;
  border-radius: 4px;
  transition: background-color 0.2s, color 0.2s;
  width: 28px; 
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.document-action-btn:hover {
  background-color: rgba(0, 0, 0, 0.05);
}
.document-action-btn.delete:hover {
  color: var(--danger);
}
.document-action-btn.download:hover {
  color: var(--accent);
}
.file-upload-area {
  border: 2px dashed var(--border);
  border-radius: 4px;
  padding: var(--space-md);
  text-align: center;
  margin-top: var(--space-md); 
  cursor: pointer;
  transition: background-color 0.2s, border-color 0.2s;
  background-color: #fdfdfd;
}
.file-upload-area:hover, .file-upload-area.dragover {
  background-color: #f0f8ff; 
  border-color: var(--accent);
}
.file-upload-area p {
  color: #666;
  font-size: 0.85rem;
  margin-bottom: var(--space-sm);
}
.file-upload-area .btn .nav-icon { 
  font-size: 1rem;
  margin-right: var(--space-xs);
}
.btn.btn-outline { 
    background-color: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: var(--space-xs) var(--space-sm);
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
}
.btn.btn-outline:hover {
    background-color: rgba(52, 152, 219, 0.1);
}
.btn .nav-icon {
  margin-right: var(--space-xs);
}
</style>