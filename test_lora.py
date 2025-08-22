import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# === [✔] Force only 1 GPU (e.g. your RTX 3050) ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use "0" or whichever index your 3050 is

# === Load Base Model ===
base_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
lora_path = "output/lora-qwen2.5-leetcode"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0},  # [✔] Explicitly maps everything to device 0 (your 3050)
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# === Chat History ===
chat_history = []

print("🤖 Qwen Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.strip().lower() == "exit":
        break

    # Append to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Format with ChatML
    formatted_prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    
    # [✔] Avoid .to("cuda") — use .to("cuda:0") instead
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode assistant response
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_reply = full_output.split("assistant")[-1].strip()

    print(f"🤖 Qwen: {assistant_reply}\n")

    # Update chat history
    chat_history.append({"role": "assistant", "content": assistant_reply})

# PS C:\LLM> python -u "c:\LLM\test_lora.py"
# Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:14<00:00,  7.23s/it]
# 🤖 Qwen Chatbot is ready! Type 'exit' to quit.

# 🧑 You: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.Merge all the linked-lists into one sorted linked-list and return it. Example 1:Input: lists = [[1,4,5],[1,3,4],[2,6]]Output: [1,1,2,3,4,4,5,6]Explanation: The linked-lists are:[  1->4->5,  1->3->4,  2->6]merging them into one sorted linked list:1->1->2->3->4->4->5->6Example 2:Input: lists = []Output: []Example 3:Input: lists = [[]]Output: [] Constraints:k == lists.length0 <= k <= 1040 <= lists[i].length <= 500-104 <= lists[i][j] <= 104lists[i] is sorted in ascending order.The sum of lists[i].length will not exceed 104.
# 🤖 Qwen: To solve the problem of merging k sorted linked lists, we can use a min-heap (priority queue) to efficiently find the smallest element among all the linked lists at any given time. Here's a Python implementation that addresses the given requirements:

# ```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def mergeKLists(self, lists: List[ListNode]) -> ListNode:
#         # Initialize a min-heap with a tuple (value, index) for each list
#         import heapq
#         heap = []
#         for i, head in enumerate(lists):
#             if head:
#                 heapq.heappush(heap, (head.val, i))

#         # Initialize the result list and its pointer
#         dummy = ListNode(0)
#         tail = dummy

#         # Process the heap until it's empty
#         while heap:
#             # Pop the smallest element from the heap
#             val, idx = heapq.heappop(heap)

#             # Add the smallest element to the result list
#             tail.next = ListNode(val)
#             tail = tail.next

#             # Move to the next node in the list from which the element was popped
#             if lists[idx].next:
#                 lists[idx] = lists[idx].next
#                 heapq.heappush(heap, (lists[idx].val, idx))

#         # Return the merged list starting from the dummy node's next
#         return dummy.next
# ```

# This solution works as follows:

# 1. **Initialization**: We initialize a min-heap (priority queue) to store the smallest elements from each linked list along with their indices.

# 2. **Populating the Heap**: We iterate through each linked list and push the first element of each non-empty list into the heap. Each element in the heap is a tuple containing the value and the index of the list from which it comes.

# 3. **Merging Process**:
#    - We repeatedly extract the smallest element from the heap.
#    - We add this element to the result list.
#    - We move to the next node in the list from which the element was extracted and push this new node into the heap if it exists.

# 4. **Return Result**: Once the heap is empty, we return the merged list starting from the dummy node's next.

# This approach ensures that we always have the smallest element available for merging, leading to an efficient O(n log k) time complexity, where n is the total number of nodes across all lists. The

# 🧑 You: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.Merge all the linked-lists into one sorted linked-list and return it. Example 1:Input: lists = [[1,4,5],[1,3,4],[2,6]]Output: [1,1,2,3,4,4,5,6]Explanation: The linked-lists are:[  1->4->5,  1->3->4,  2->6]merging them into one sorted linked list:1->1->2->3->4->4->5->6Example 2:Input: lists = []Output: []Example 3:Input: lists = [[]]Output: [] Constraints:k == lists.length0 <= k <= 1040 <= lists[i].length <= 500-104 <= lists[i][j] <= 104lists[i] is sorted in ascending order.The sum of lists[i].length will not exceed 104. give
#  cpp solution
# 🤖 Qwen: Certainly! Below is a C++ implementation of the solution to merge k sorted linked lists:

# ```cpp
# #include <vector>
# #include <queue>
# using namespace std;

# struct ListNode {
#     int val;
#     ListNode* next;
#     ListNode(int x) : val(x), next(NULL) {}
# };

# class Solution {
# public:
#     ListNode* mergeKLists(vector<ListNode*>& lists) {
#         // Priority queue to store the current minimum element and its list index
#         priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

#         // Initialize the priority queue with the first element of each list
#         for (int i = 0; i < lists.size(); ++i) {
#             if (lists[i]) {
#                 pq.push({lists[i]->val, i});
#             }
#         }

#         // Dummy node to start the merged list
#         ListNode dummy(0);
#         ListNode* tail = &dummy;

#         // Merge the lists using the priority queue
#         while (!pq.empty()) {
#             // Extract the minimum element
#             auto [val, idx] = pq.top();
#             pq.pop();

#             // Append the minimum element to the result list
#             tail->next = new ListNode(val);
#             tail = tail->next;

#             // Move to the next element in the list
#             if (lists[idx]->next) {
#                 lists[idx] = lists[idx]->next;
#                 pq.push({lists[idx]->val, idx});
#             }
#         }

#         // Return the merged list starting from the dummy node's next
#         return dummy.next;
#     }
# };
# ```

# ### Explanation:

# 1. **ListNode Structure**: We define a `ListNode` structure to represent the nodes of the linked lists.

# 2. **Priority Queue**: We use a priority queue (`std::priority_queue`) to keep track of the minimum element among all the linked lists. The priority queue is implemented as a min-heap, so the smallest element is always at the top.

# 3. **Initialization**: We initialize the priority queue with the first element of each non-empty list. This ensures that the smallest element is always available for merging.

# 4. **Merging Process**:
#    - We repeatedly extract the minimum element from the priority queue.
#    - We append this element to the result list.
#    - We move to the next element in the list from which the element was extracted and push this new element into the priority queue if it exists.

# 5. **Return Result**: Once the priority





#torch version = 2.7.1+cu118
#pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
