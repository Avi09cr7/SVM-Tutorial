import React from "react";
import Quiz from "react-quiz-component";
import { quiz } from "../../utils/quiz";

function AppQuiz() {
  const setQuizResult = (obj) => {
    console.log({ obj });
    // YOUR LOGIC GOES HERE
    // alert(`You got ${obj.correctPoints} / ${obj.totalPoints}`);
  };
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Quiz
        quiz={quiz}
        showDefaultResult={true}
        shuffle={true}
        onComplete={setQuizResult}
      />
    </div>
  );
}

export default AppQuiz;
