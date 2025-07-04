import streamlit as st
from PIL import Image
from agent import agent
import matplotlib.pyplot as plt
from tools.plot_tool import get_last_figure

st.set_page_config(page_title="📊 InsightBot", layout="wide")
st.title("📊 InsightBot: Ask Data Questions")

# Keep history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask InsightBot about your data...")

if user_input:
    st.chat_message("user").write(user_input)
    with st.spinner("🤖 Thinking..."):
        try:
            # Run agent
            output = agent.run(user_input)
            
            # Check if a figure was generated
            figure = get_last_figure()
            
            # Debug: Show what type of output we got
            st.write(f"🔍 Debug: Output type is {type(output)}")
            st.write(f"🔍 Debug: Figure available: {figure is not None}")
            st.write(f"🔍 Debug: Output length: {len(str(output)) if output else 0} characters")

            # Display the output
            with st.chat_message("assistant"):
                # First check if we have a figure to display
                if figure is not None:
                    st.write("✅ Displaying generated visualization")
                    st.pyplot(figure)
                    plt.close(figure)  # Close the figure to prevent memory leaks
                    # Also show text response if available
                    if output and isinstance(output, str) and len(output) > 50:
                        st.markdown("### 📝 Analysis:")
                        st.markdown(output)
                    # Store both in history
                    st.session_state.history.append((user_input, (figure, output)))
                elif isinstance(output, plt.Figure):
                    st.write("✅ Displaying matplotlib figure from output")
                    st.pyplot(output)
                    plt.close(output)  # Close the figure to prevent memory leaks
                    st.session_state.history.append((user_input, output))
                elif isinstance(output, Image.Image):
                    st.write("✅ Displaying PIL image")
                    st.image(output, caption="📈 Generated Visualization")
                    st.session_state.history.append((user_input, output))
                else:
                    st.markdown("### 📝 Analysis:")
                    # Use markdown for better formatting
                    st.markdown(output)
                    st.session_state.history.append((user_input, output))

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(f"🔍 Full traceback: {traceback.format_exc()}")

# Display chat history
with st.expander("📜 Chat History"):
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**{i+1}. Q:** {q}")
        if isinstance(a, tuple):  # Handle (figure, text) tuple
            fig, text = a
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            if text:
                st.markdown(text)
        elif isinstance(a, plt.Figure):
            st.pyplot(a)
            plt.close(a)  # Close the figure to prevent memory leaks
        elif isinstance(a, Image.Image):
            st.image(a)
        else:
            st.markdown(str(a))

st.markdown("---")
st.caption("Built with LangChain, Claude, and Streamlit")
